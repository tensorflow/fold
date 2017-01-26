<!-- This file is machine generated: DO NOT EDIT! -->

# `class tensorflow::fold::WeaverOpBase`



` WeaverOpBase ` is a base class for writing TensorFlow ops kernels that schedule ops for Loom.

Operations created as subclasses of ` WeaverOpBase ` should be registered with the `REGISTER_WEAVER_OP` macro. For example, ` DeserializingWeaverOp ` is registered using:

```c++
REGISTER_WEAVER_OP("DeserializingWeaver").Input("weaver_messages: string");
```

And

```c++
REGISTER_KERNEL_BUILDER(
    Name("DeserializingWeaver").Device(tensorflow::DEVICE_CPU),
    DeserializingWeaverOp);
```

###Member Details

<a name="tensorflow_fold_WeaverOpBase_WeaverOpBase"></a>
#### `tensorflow::fold::WeaverOpBase::WeaverOpBase(tensorflow::OpKernelConstruction *c)`



Reads the `metadata`, `constant_types`, and `num_types_shapes` attributes and makes sure they&apos;re consistent. Dies if they&apos;re not.

<a name="virtual_tensorflow_Status_tensorflow_fold_WeaverOpBase_Weave"></a>
#### `virtual tensorflow::Status tensorflow::fold::WeaverOpBase::Weave(tensorflow::OpKernelContext *c, Weaver *weaver)=0`



Weave is a virtual method, to be subclassed. Weave&apos;s responsibility is to read the ops inputs and use the weaver to schedule LoomOps to be executed on the loom. `Weave` should not call ` Weaver::Finalize `.

<a name="void_tensorflow_fold_WeaverOpBase_Compute"></a>
#### `void tensorflow::fold::WeaverOpBase::Compute(tensorflow::OpKernelContext *c) override`



Dispatches to `Weave` to build a ` Weaver `, which is then used to build the wiring diagram and constant tensors that the loom needs.
