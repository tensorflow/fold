/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This is an example model which implements a Tree RNN.
//
// A Tree RNN is a Recursive Neural Network that operates over trees.
// It is a generalization of the more common Recurrent Neural Network (RNN)
// which operates over sequences.
//
// Given an input sequence, a sequence RNN traverses the sequence, and computes
// a state for each element. The state for a given element is computed by
// applying an RNN cell, which takes the element data, and the previous
// state, as inputs.
//
// Similarly, a Tree-RNN recursively traverses a tree, and computes a state
// for each node. The state is computed by applying a Tree-RNN cell, which
// takes the data for the node, and state of any child nodes, as inputs. The
// only difference is that a Tree-RNN cell must combine states from multiple
// children, instead of just one predecessor.
//
// Most RNN cell types (simple RNN, LSTM, GRU, etc.) can be generalized to
// work with trees.  (E.g. https://arxiv.org/abs/1503.00075).

#include "tensorflow_fold/llgtm/backend/eigen_evaluator_client.h"
#include "tensorflow_fold/llgtm/examples/parsetree.h"
#include "tensorflow_fold/llgtm/llgtm.h"


BEGIN_COMMAND_LINE_FLAGS;
DEFINE_FLAG(double, learning_rate, 0.01, "Learning rate");
DEFINE_FLAG(int, num_steps, 1000, "Number of steps to train");
DEFINE_FLAG(int, num_hidden, 128, "Number of hidden units for RNN");
DEFINE_FLAG(int, num_chars, 128, "Number of characters in character set");
DEFINE_FLAG(int, embedding_size, 16, "Embedding size for characters");


using llgtm::Dimensions;
using llgtm::FullyConnectedLayer;
using llgtm::Graph;
using llgtm::GraphEvaluator;
using llgtm::MomentumTrainer;
using llgtm::Tensor;
using llgtm::UniformRandomInitializer;
using llgtm::Variable;
using llgtm::VariableSet;

using llgtm::parse_trees::TreeNode;
using llgtm::parse_trees::LeafNode;
using llgtm::parse_trees::PhraseNode;
using llgtm::parse_trees::Leaf;
using llgtm::parse_trees::Phrase;


// Select Eigen as the evaluation backend.
// Client code must also include eigen_evaluator_client.h,
// and link against llgtm_eigen.
using GraphEvaluatorImpl = llgtm::EigenEvaluator;

// The Tree RNN model runs a character RNN over each word, to embed the
// words into continuous vector space. It then runs a Tree-RNN, to recursively
// embed each phrase into vector space. Lastly, it feeds the output of the
// sentence (top-level phrase) into a softmax classifier.
class TreeRnnModel {
 public:
  explicit TreeRnnModel(GraphEvaluator* evaluator)
      : evaluator_(evaluator), model_(evaluator),

        // Set hyper-parameters from command-line arguments.
        learning_rate_(GET_CL_FLAG(learning_rate)),
        num_steps_(GET_CL_FLAG(num_steps)),
        num_hidden_(GET_CL_FLAG(num_hidden)),
        num_chars_(GET_CL_FLAG(num_chars)),
        embedding_size_(GET_CL_FLAG(embedding_size)),
        num_labels_(2),

        // Table which embeds each character into continuous vector space.
        embedding_table_(model_.NewVariable<float>(
            "char_embedding_table",
            Dimensions(num_chars_, embedding_size_),
            /*parent=*/ nullptr,
            UniformRandomInitializer<float>())),
        // Simple RNN cell for the character RNN.
        // Relu is unstable on long RNNs, so we use sigmoid instead.
        word_layer_(absl::make_unique<FullyConnectedLayer>(
            model_.NewNameSpace("word_cell"),
            /*num_hidden=*/ num_hidden_,
            FullyConnectedLayer::kSigmoid)),
        // Simple RNN cell for the tree RNN.
        // Trees usually aren't very deep, so we can use relu.
        phrase_layer_(absl::make_unique<FullyConnectedLayer>(
            model_.NewNameSpace("phrase_cell"),
            /*num_hidden=*/ num_hidden_,
            FullyConnectedLayer::kRelu)),
        // Softmax classifier.
        logits_layer_(absl::make_unique<FullyConnectedLayer>(
            model_.NewNameSpace("logits"),
            /*num_hidden=*/ num_labels_,
            FullyConnectedLayer::kLinear))
  {
    CreateExampleData();
  }

  // Trains this model on a corpus of examples.
  // Training will update the parameters (i.e. Variables) of this model.
  void Train() {
    MomentumTrainer trainer(learning_rate_);
    for (int step = 0; step < num_steps_; ++step) {
      LOG(INFO) << "Step: " << step;

      // Create an empty graph.
      Graph graph = evaluator_->NewGraph();

      // Build the graph, unrolling the RNN over the input sequence.
      auto loss = TrainStep(&graph);

      // Backpropogation pass: symbolically differentiate the graph with
      // respect to model parameters, and add operations which update parameters
      // to the graph.
      //
      // The following statement is shorthand for the following:
      //   Gradients grads(&model_);
      //   g.ComputeGradients(&grads, loss);
      //   trainer.ApplyGradients(grads);
      trainer.ComputeAndApplyGradients(&graph, &model_, loss);

      // Evaluate the graph, including the parameter updates.
      // No tensor computations are done until now.
      // (Note that the log statements come with timestamps, which give us
      // rough timing info for each step.)
      LOG(INFO) << "Evaluate...";
      graph.Eval();

      float loss_v = *loss.result_data();
      LOG(INFO) << "Loss: " << loss_v;

      // The graph is destroyed here.
    }
  }

 private:
  // Runs a character RNN (sequence model) over the string in a leaf node.
  // By convention, the short name "g" is used for the graph in graph
  // construction code.
  Tensor<float> TraverseLeafNode(Graph* g, const LeafNode* node) {
    // Embedding table for ASCII characters.
    auto table = g->Variable(embedding_table_);
    // Initial hidden state vector for the RNN.
    Tensor<float> state = g->Zeros<float>(Dimensions(1, num_hidden_));

    for (char c : node->word()) {
      Tensor<int32_t> idx = g->ConstantFromScalar<int32_t>(Dimensions(1), c);
      auto embedding = g->Gather(idx, table);   // Embed the character.
      auto x = g->Concat(state, embedding, 1);  // Combine character and state.
      state = (*word_layer_)(g, x);             // Compute the next state.
    }
    return state;
  }

  // Runs a Tree-RNN over a non-terminal node, which traverses the tree
  // recursively.
  Tensor<float> TraversePhraseNode(Graph* g, const PhraseNode* node) {
    // Use a recurrent NN to combine states from children.
    Tensor<float> state = g->Zeros<float>(Dimensions(1, num_hidden_));
    for (const auto& subnode : node->sub_nodes()) {
      auto subphrase = TraverseNode(g, subnode.get());
      auto x = g->Concat(state, subphrase, 1);  // Input datum to the RNN.
      state = (*phrase_layer_)(g, x);           // Compute next state.
    }
    return state;
  }

  // Traverses the tree, dispatching on the type of node.
  Tensor<float> TraverseNode(Graph* g, TreeNode* node) {
    CHECK(node);
    return node->SwitchOnType(
      [=](LeafNode* leaf) {
        return TraverseLeafNode(g, leaf);
      },
      [=](PhraseNode* phrase) {
        return TraversePhraseNode(g, phrase);
      }
    );
  }

  // Takes a tree as input, runs the Tree RNN over the nodes, and attempts to
  // classify the tree.  Returns the classification loss.
  // Tensor operations are added to the Graph g.
  Tensor<float> ComputeLoss(Graph* g, TreeNode* node, int node_label) {
    auto state = TraverseNode(g, node);
    auto logits = (*logits_layer_)(g, state);
    auto label = g->ConstantFromScalar<int32_t>(Dimensions(1), node_label);
    auto probabilities = g->Softmax(logits);
    auto sm_loss = g->SparseCrossEntropyLoss(label, probabilities);

    // For batch size 1, we can just do a reshape to get a scalar loss.
    return g->Reshape(sm_loss, Dimensions());
  }

  // Trains the Tree RNN for one step.
  // Classifies each example in the example data.
  Tensor<float> TrainStep(Graph* g) {
    Tensor<float> loss = g->Zeros<float>(Dimensions());
    int i = 0;
    for (auto& node : example_data_) {
      DCHECK_LT(i, num_labels_);
      loss = g->Add(loss, ComputeLoss(g, node.get(), /*node_label=*/ i));
      ++i;
    }
    return loss;
  }

  // Creates some training data.
  // This simple version simply memorizes two difference sentences, and
  // learns to distinguish between the two.
  void CreateExampleData() {
    example_data_.emplace_back(
        Phrase(Phrase(Leaf("The"),
                      Phrase(Leaf("quick"),
                             Phrase(Leaf("brown"),
                                    Leaf("fox")))),
               Leaf("jumped"),
               Phrase(Leaf("over"),
                      Phrase(Leaf("the"),
                             Phrase(Leaf("lazy"),
                                    Leaf("dog"))))));

    example_data_.emplace_back(
        Phrase(Phrase(Leaf("The"),
                      Phrase(Leaf("bouncy"),
                             Phrase(Leaf("brown"),
                                    Leaf("cow")))),
               Leaf("jumped"),
               Phrase(Leaf("over"),
                      Phrase(Leaf("the"),
                             Phrase(Leaf("bright"),
                                    Leaf("moon"))))));
  }

  GraphEvaluator* const evaluator_;
  VariableSet model_;
  std::vector<std::unique_ptr<TreeNode>> example_data_;

  // Model hyper-parameters.
  const float learning_rate_;
  const int num_steps_;
  const int num_hidden_;
  const int num_chars_;
  const int embedding_size_;
  const int num_labels_;

  // Model parameters.  Variables are created and owned by model_.
  Variable<float>* const embedding_table_;
  const std::unique_ptr<FullyConnectedLayer> word_layer_;
  const std::unique_ptr<FullyConnectedLayer> phrase_layer_;
  const std::unique_ptr<FullyConnectedLayer> logits_layer_;
};


int main(int argc, char** argv) {
  PARSE_COMMAND_LINE_FLAGS(argc, argv);

  // The GraphEvaluator must persist for the lifetime of the model.
  GraphEvaluatorImpl evaluator;
  TreeRnnModel rnn_model(&evaluator);

  rnn_model.Train();

  return 0;
}
