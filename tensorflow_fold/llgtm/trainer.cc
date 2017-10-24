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

#include "tensorflow_fold/llgtm/trainer.h"

#include "tensorflow_fold/llgtm/graph.h"
#include "tensorflow_fold/llgtm/graph_evaluator.h"
#include "tensorflow_fold/llgtm/tensor_ops_impl.h"
#include "tensorflow_fold/llgtm/variable_initializers.h"
#include "absl/memory/memory.h"

namespace llgtm {

void Trainer::CheckInitialize(const Gradients& grads) {
  if (variables_ == nullptr) {
    Initialize(grads);
  } else if (variables_ != grads.variable_set()) {
    LOG(FATAL) << "Trainer can only be used with one VariableSet.";
  }
}

void Trainer::Initialize(const Gradients& grads) {
  variables_ = grads.variable_set();
  for (VariableBase& var : *variables()) {
    if (var.dtype() == kDTfloat64) {
      LOG(WARNING) << "Warning: gradients for double precision floating point "
                   << "variables are not currently supported. (" << var.name()
                   << ")";
    }
  }
}

void Trainer::ComputeAndApplyGradients(Graph* graph, VariableSet* model,
                                       Tensor<float> loss) {
  CHECK_NOTNULL(graph);
  Gradients grads(model);
  graph->ComputeGradients(&grads, std::move(loss));
  ApplyGradients(grads);
}


void SGDTrainer::ApplyGradients(const Gradients& grads) {
  CheckInitialize(grads);

  Graph* g = grads.graph();
  for (VariableBase& var_base : *variables()) {
    // TODO(delesley): Support gradients for half and double precision floats.
    if (var_base.dtype() != kDTfloat32) continue;
    if (grads.has_gradient(&var_base)) {
      Variable<float>* var = var_base.as<float>();

      // Use negative learning rate because we need to subtract the gradient.
      auto lrate = g->ConstantFromScalar<float>(var->dimensions(),
                                                -learning_rate_);
      g->AssignAdd(var, g->Multiply(grads.Gradient(var), lrate));
    }
  }
}


void MomentumTrainer::Initialize(const Gradients& grads) {
  Trainer::Initialize(grads);

  // Set up a VariableSet and namespaces for momentum variables.
  momentum_var_map_.resize(variables()->size(), nullptr);
  momentum_variables_ =
      absl::make_unique<VariableSet>(variables()->evaluator(), "momentum");
  momentum_variables_->CopyNameSpaces(*variables());

  // Instantiate a momentum variable for each trainable variable.
  for (VariableBase& var : *variables()) {
    // Only store momentum for floating point variables.
    // TODO(delesley): Support gradients for half and double precision floats.
    if (var.dtype() != kDTfloat32) continue;

    // CopyNameSpaces yields a 1:1 map over namespace ids.
    VarNameSpace* shadow_parent =
        momentum_variables_->get_namespace(var.parent()->id());
    Variable<float>* shadow_var =
        momentum_variables_->NewVariable<float>(var.name(), var.dimensions(),
                                                shadow_parent,
                                                ZerosInitializer<float>());
    momentum_var_map_[var.id()] = shadow_var;
  }
}

void MomentumTrainer::ApplyGradients(const Gradients& grads) {
  CheckInitialize(grads);

  // Momentum starts at zero, and gradually approaches the requested value.
  // This gives the learning time to settle before momentum kicks in.
  active_momentum_ = (active_momentum_ * (1.0f - kMomentumRampRate)) +
                     (momentum_ * kMomentumRampRate);

  Graph* g = grads.graph();
  for (VariableBase& var_base : *variables()) {
    if (var_base.dtype() != kDTfloat32) continue;

    Variable<float>* var = var_base.as<float>();
    Variable<float>* momentum_var = momentum_var_map_[var->id()];
    if (grads.has_gradient(var)) {
      // Implements the momentum algorithm:
      // momentum = momentum*momentum_decay_rate + gradient*learning_rate
      // variables -= momentum

      // Use negative learning rate because we need to subtract the gradient.
      auto lrate = g->ConstantFromScalar<float>(var->dimensions(),
                                                -learning_rate_);
      auto mrate = g->ConstantFromScalar<float>(var->dimensions(),
                                                active_momentum_ - 1.0f);

      auto momentum_val = g->Variable(momentum_var);
      auto momentum_delta = g->Add(g->Multiply(momentum_val, mrate),
                                   g->Multiply(grads.Gradient(var), lrate));

      g->AssignAdd(momentum_var, momentum_delta);
      // momentum_val has now holds the new value of momentum.
      g->AssignAdd(var, momentum_val);
    }
  }
}

}  // namespace llgtm
