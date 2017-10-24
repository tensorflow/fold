// This file lists all node types that are templatized by data type.
// Nodes that are not templatized by data type (DT) are handled separately.

// Since the Eigen backend is the reference backend implementation, it must
// support all LLGTM nodes types.

// This header file is used to generate source code for all node types in
// various places without having to write down all node types over and over.
// It should be used as follows:
// 1. Define a preprocessor macro LLGTM_NODE_DEFINITION(NODE) expanding to the
//    source code that should be generated for every node.
// 2. (Optional) Define a preprocessor macro LLGTM_NODE_DEFINITION_FP, for
//    for node types that are floating-point only.  If unspecified, this will
//    default to LLGTM_NODE_DEFINITION.
// 3. Include llgtm_nodes.inc. Note: It can be included multiple times.
// 4. Undefine LLGTM_DEFINITION_NODE
// See also "Textual Headers" in go/totw/127.

// The main use case of this file is to force template expansion of all node
// types/classes.

#ifndef LLGTM_NODE_DEFINITION_FP
#define LLGTM_NODE_DEFINITION_FP(NODE) LLGTM_NODE_DEFINITION(NODE)
#endif

LLGTM_NODE_DEFINITION(nodes::Add);
LLGTM_NODE_DEFINITION(nodes::AssignAdd);
LLGTM_NODE_DEFINITION(nodes::Broadcast);
LLGTM_NODE_DEFINITION(nodes::Concat);
LLGTM_NODE_DEFINITION(nodes::ConstantFromScalar);
LLGTM_NODE_DEFINITION(nodes::CopyToDevice);
LLGTM_NODE_DEFINITION(nodes::Gather);
LLGTM_NODE_DEFINITION(nodes::Matmul);
LLGTM_NODE_DEFINITION(nodes::Multiply);
LLGTM_NODE_DEFINITION(nodes::Negative);
LLGTM_NODE_DEFINITION_FP(nodes::NormalRandom);
LLGTM_NODE_DEFINITION_FP(nodes::Reciprocal);
LLGTM_NODE_DEFINITION(nodes::Relu);
LLGTM_NODE_DEFINITION(nodes::ReluGrad);
LLGTM_NODE_DEFINITION(nodes::ReduceSum);
LLGTM_NODE_DEFINITION(nodes::Reshape);
LLGTM_NODE_DEFINITION(nodes::Scatter);
LLGTM_NODE_DEFINITION_FP(nodes::Sigmoid);
LLGTM_NODE_DEFINITION_FP(nodes::Softmax);
LLGTM_NODE_DEFINITION_FP(nodes::SoftmaxCrossEntropy);
LLGTM_NODE_DEFINITION_FP(nodes::SoftmaxSparseCrossEntropy);
LLGTM_NODE_DEFINITION_FP(nodes::SoftmaxSparseCrossEntropyGrad);
LLGTM_NODE_DEFINITION(nodes::Split);
LLGTM_NODE_DEFINITION_FP(nodes::Tanh);
LLGTM_NODE_DEFINITION(nodes::TensorValue);
LLGTM_NODE_DEFINITION(nodes::TensorVariable);
LLGTM_NODE_DEFINITION(nodes::Transpose);
LLGTM_NODE_DEFINITION_FP(nodes::UniformRandom);
LLGTM_NODE_DEFINITION(nodes::Zeros);

#undef LLGTM_NODE_DEFINITION_FP

// Not listed here (due to different template parameters):
// LLGTM_NODE_DEFINITION(nodes::GetOutput)
// LLGTM_NODE_DEFINITION(nodes::ConstantFromFunction)
