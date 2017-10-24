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

// Interface for training neural networks.
// A Trainer applies gradients to a VariableSet.

#ifndef TENSORFLOW_FOLD_LLGTM_TRAINER_H_
#define TENSORFLOW_FOLD_LLGTM_TRAINER_H_

#include "tensorflow_fold/llgtm/gradients.h"
#include "tensorflow_fold/llgtm/tensor.h"
#include "tensorflow_fold/llgtm/tensor_ops.h"

namespace llgtm {

// Base class for Trainers.
class Trainer {
 public:
  Trainer() {}
  Trainer(const Trainer&) = delete;
  Trainer(Trainer&&) = delete;
  virtual ~Trainer() {}

  Trainer& operator=(const Trainer&) = delete;
  Trainer& operator=(Trainer&&) = delete;

  // Adds nodes to the graph which apply the gradients to the variables.
  // The model will not be updated until Graph.Eval() is called.
  virtual void ApplyGradients(const Gradients& grads) = 0;

  // Computes gradients for the given loss, and add nodes to the graph which
  // applies them to the model.  The model will not be updated until
  // Graph.Eval() is called.  It will fail if graph or model are nullptr.
  void ComputeAndApplyGradients(Graph* graph, VariableSet* model,
                                Tensor<float> loss);

  VariableSet* variables() { return variables_; }

 protected:
  // Initializes the trainer, which happens on the first call to ApplyGradients.
  // Derived classes may override this method to allocate additional state
  // associated with training (e.g. momentum). The overridden version should
  // call the base class version.
  virtual void Initialize(const Gradients& grads);

  // Initializes the trainer if it has not been initialized already.
  // Derived classes must call this method at the start of ApplyGradients.
  void CheckInitialize(const Gradients& grads);

 private:
  VariableSet* variables_ = nullptr;
};


// Trainer which implements stochastic gradient descent.
class SGDTrainer : public Trainer {
 public:
  SGDTrainer() = delete;
  explicit SGDTrainer(float learning_rate) : learning_rate_(learning_rate) {}

  float learning_rate() const { return learning_rate_; }

  void ApplyGradients(const Gradients& grads) override;

 private:
  float learning_rate_;
};


// Trainer which implements the stochastic gradient descent with momentum.
class MomentumTrainer : public Trainer {
 public:
  MomentumTrainer() = delete;

  // Creates a new MomentumTrainer.
  explicit MomentumTrainer(float learning_rate, float momentum = 0.9f)
    : learning_rate_(learning_rate), momentum_(momentum) {}

  float learning_rate() const { return learning_rate_; }
  float momentum() const { return momentum_; }

  // For every differentiable variable in the model, MomentumTrainer
  // creates another variable of the same shape to track its momentum.
  VariableSet* momentum_variables() { return momentum_variables_.get(); }

  Variable<float>* momentum_variable(VariableBase* var) {
    DCHECK_EQ(var->variable_set(), variables());
    DCHECK_LE(var->id(), momentum_var_map_.size());
    return momentum_var_map_[var->id()];
  }

  void ApplyGradients(const Gradients& grads) override;

 protected:
  void Initialize(const Gradients& grads) override;

 private:
  // The rate at which the momentum ramps up from 0 to its proper value.
  static constexpr float kMomentumRampRate = 0.05f;

  std::unique_ptr<VariableSet>  momentum_variables_;
  std::vector<Variable<float>*> momentum_var_map_;

  float learning_rate_;
  float momentum_;
  float active_momentum_ = 0.0f;
};

}  // namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_TRAINER_H_
