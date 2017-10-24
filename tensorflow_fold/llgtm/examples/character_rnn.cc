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

// This is an example model which runs a character RNN over a string.
// A good description of character RNNs can be found at:
// http://karpathy.github.io/2015/05/21/rnn-effectiveness/

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow_fold/llgtm/backend/eigen_evaluator_client.h"
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


// Select Eigen as the evaluation backend.
// Client code must also include eigen_evaluator_client.h,
// and link against llgtm_eigen.
using GraphEvaluatorImpl = llgtm::EigenEvaluator;


class CharacterRnnModel {
 public:
  explicit CharacterRnnModel(GraphEvaluator* evaluator)
      : evaluator_(evaluator), model_(evaluator),

        // Set hyper-parameters from command-line arguments.
        learning_rate_(GET_CL_FLAG(learning_rate)),
        num_steps_(GET_CL_FLAG(num_steps)),
        num_hidden_(GET_CL_FLAG(num_hidden)),
        num_chars_(GET_CL_FLAG(num_chars)),
        embedding_size_(GET_CL_FLAG(embedding_size)),

        // Table which embeds each character into continuous vector space.
        embedding_table_(model_.NewVariable<float>(
            "char_embedding_table",
            Dimensions(num_chars_, embedding_size_),
            /*parent=*/ nullptr,
            UniformRandomInitializer<float>())),
        // Simple RNN cell to get the next state.
        // Relu is unstable on long RNNs, so we use sigmoid instead.
        rnn_layer_(absl::make_unique<FullyConnectedLayer>(
            model_.NewNameSpace("rnn_cell"),
            num_hidden_,
            FullyConnectedLayer::kSigmoid)),
        // Predict next character from current state.
        logits_layer_(absl::make_unique<FullyConnectedLayer>(
            model_.NewNameSpace("logits"),
            num_hidden_,
            FullyConnectedLayer::kLinear))
  {}

  // Trains this model on a corpus of examples.
  // Training will update the parameters (i.e. Variables) of this model.
  void Train() {
    // TODO(delesley):  Add support for a proper corpus.
    // This version just memorizes a sentence.
    absl::string_view datum = "The quick brown fox jumped over the lazy dog.";

    MomentumTrainer trainer(learning_rate_);
    for (int step = 0; step < num_steps_; ++step) {
      LOG(INFO) << "Step: " << step;

      // Create an empty graph.
      Graph graph = evaluator_->NewGraph();

      // Build the graph, unrolling the RNN over the input sequence.
      auto loss = TrainStep(&graph, datum);

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
      LOG(INFO) << "Evaluate...";  // Get rough timing for build vs. eval.
      graph.Eval();

      float loss_v = *loss.result_data();
      LOG(INFO) << "Loss: " << loss_v;

      // The graph is destroyed here.
    }
  }

  // TODO(delesley): Add additional methods to save and restore model from
  // disk, once that capability has been added to LLGTM.

 private:
  // Takes a string as input, runs an RNN to predict each character in the
  // string, and returns a scalar loss as output. Tensor operations are added
  // to the Graph g.  By convention, the short name "g" is used for the graph
  // in graph construction code.
  Tensor<float> TrainStep(Graph* g, absl::string_view str) {
    CHECK_NOTNULL(g);

    Tensor<float> loss = g->Zeros<float>(Dimensions());
    // Nothing to predict, so the loss is zero.
    if (str.empty()) return loss;

    const int batch_size = 1;
    Dimensions character_dims(batch_size);

    // Embedding table for ASCII characters.
    auto table = g->Variable(embedding_table_);
    // Initial hidden state vector for the RNN
    Tensor<float> state = g->Zeros<float>(Dimensions(batch_size, num_hidden_));
    // Convert first character into an integer tensor.  (str is non-empty).
    Tensor<int32_t> current_char =
        g->ConstantFromScalar<int32_t>(character_dims, str[0]);

    for (int i = 0, str_size = str.size(); i < str_size; ++i) {
      // Implement simple predictive RNN.
      auto embedding = g->Gather(current_char, table);  // Embed character.
      auto x = g->Concat(state, embedding, 1);
      state = (*rnn_layer_)(g, x);               // Compute next state.
      auto logits = (*logits_layer_)(g, state);  // Predict char from state.
      auto probabilities = g->Softmax(logits);

      // Loss for character prediction, or predict 0 for end-of-string.
      char next_c = i < (str_size - 1) ? str[i + 1] : 0;
      auto next_char = g->ConstantFromScalar<int32_t>(character_dims, next_c);
      auto sm_loss = g->SparseCrossEntropyLoss(next_char, probabilities);

      // Sum the prediction losses for the whole sequence.
      // For batch size 1, we can just do a reshape to get a scalar loss.
      loss = g->Add(loss, g->Reshape(sm_loss, Dimensions()));
      current_char = next_char;
    }
    return loss;
  }

  GraphEvaluator* const evaluator_;
  VariableSet model_;

  // Model hyper-parameters.
  const float learning_rate_;
  const int num_steps_;
  const int num_hidden_;
  const int num_chars_;
  const int embedding_size_;

  // Model parameters.  Variables are created and owned by model_.
  Variable<float>* const embedding_table_;
  const std::unique_ptr<FullyConnectedLayer> rnn_layer_;
  const std::unique_ptr<FullyConnectedLayer> logits_layer_;
};


int main(int argc, char** argv) {
  PARSE_COMMAND_LINE_FLAGS(argc, argv);

  // The GraphEvaluator must persist for the lifetime of the model.
  GraphEvaluatorImpl evaluator;
  CharacterRnnModel rnn_model(&evaluator);

  rnn_model.Train();

  return 0;
}
