<!-- This file is machine generated: DO NOT EDIT! -->

# `class tensorflow::fold::WeaverOpBase`



WeaverOpBase is a base class for writing TensorFlow ops kernels that schedule ops for Loom.

Operations created as subclasses of WeaverOpBase should be registered with the REGISTER_WEAVER_OP macro. For example, DeserializingWeaverOp is registered using:

REGISTER_WEAVER_OP("DeserializingWeaver") .Input("weaver_messages: string");

And

REGISTER_KERNEL_BUILDER( Name("DeserializingWeaver").Device(tensorflow::DEVICE_CPU), DeserializingWeaverOp );

###Member Details

#### `tensorflow::fold::WeaverOpBase::WeaverOpBase(tensorflow::OpKernelConstruction *c)` {#tensorflow_fold_WeaverOpBase_WeaverOpBase}



Reads the metadata, constant_types, and num_types_shapes attributes and makes sure they&apos;re consistent. Dies if they&apos;re not.

#### `virtual tensorflow::Status tensorflow::fold::WeaverOpBase::Weave(tensorflow::OpKernelContext *c, Weaver *weaver)=0` {#virtual_tensorflow_Status_tensorflow_fold_WeaverOpBase_Weave}



Weave is a virtual method, to be subclassed. Weave&apos;s responsibility is to read the ops inputs and use the weaver to schedule LoomOps to be executed on the loom. `Weave` should not call ` Weaver::Finalize `.

#### `void tensorflow::fold::WeaverOpBase::Compute(tensorflow::OpKernelContext *c) override` {#void_tensorflow_fold_WeaverOpBase_Compute}



Dispatches to Weave to build a Weaver , which is then used to build the wiring diagram and constant tensors that the loom needs.
