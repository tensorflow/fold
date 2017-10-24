// This file lists all node types that are supported by the TF backend.
// Nodes that are not templatized by data type (DT) are handled separately.

// See llgtm_nodes.inc for detailed explanations.

LLGTM_TF_KERNEL_DEFINITION(nodes::Add);
LLGTM_TF_KERNEL_DEFINITION(nodes::ConstantFromScalar);
LLGTM_TF_KERNEL_DEFINITION(nodes::Matmul);
LLGTM_TF_KERNEL_DEFINITION(nodes::Multiply);
LLGTM_TF_KERNEL_DEFINITION(nodes::Negative);
LLGTM_TF_KERNEL_DEFINITION(nodes::NormalRandom);
LLGTM_TF_KERNEL_DEFINITION(nodes::Reciprocal);
LLGTM_TF_KERNEL_DEFINITION(nodes::ReduceSum);
LLGTM_TF_KERNEL_DEFINITION(nodes::Relu);
LLGTM_TF_KERNEL_DEFINITION(nodes::ReluGrad);
LLGTM_TF_KERNEL_DEFINITION(nodes::Reshape);
LLGTM_TF_KERNEL_DEFINITION(nodes::Sigmoid);
LLGTM_TF_KERNEL_DEFINITION(nodes::Tanh);
LLGTM_TF_KERNEL_DEFINITION(nodes::Transpose);
LLGTM_TF_KERNEL_DEFINITION(nodes::UniformRandom);
LLGTM_TF_KERNEL_DEFINITION(nodes::Zeros);
