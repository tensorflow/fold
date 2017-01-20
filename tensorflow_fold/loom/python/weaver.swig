
%{
#include <vector>
#include "tensorflow_fold/loom/weaver.h"
#include "tensorflow_fold/loom/platform.h"

using tensorflow::int64;
using tensorflow::int32;
using tensorflow::string;

  template<class T>
      bool _PyObjAs(PyObject *pystr, T* cstr) {
    T::undefined;  // You need to define specialization _PyObjAs<T>
    return false;
  }

  template<class T>
      PyObject *_PyObjFrom(const T& c) {
    T::undefined;  // You need to define specialization _PyObjFrom<T>
    return NULL;
  }
#ifdef HAS_GLOBAL_STRING
  template<>
      bool _PyObjAs(PyObject *pystr, ::string* cstr) {
    char *buf;
    Py_ssize_t len;
    if (PyBytes_AsStringAndSize(pystr, &buf, &len) == -1) return false;
    if (cstr) cstr->assign(buf, len);
    return true;
  }
#endif
  template<>
      bool _PyObjAs(PyObject *pystr, std::string* cstr) {
    char *buf;
    Py_ssize_t len;
    if (PyBytes_AsStringAndSize(pystr, &buf, &len) == -1) return false;
    if (cstr) cstr->assign(buf, len);
    return true;
  }
#ifdef HAS_GLOBAL_STRING
  template<>
      PyObject* _PyObjFrom(const ::string& c) {
    return PyBytes_FromStringAndSize(c.data(), c.size());
  }
#endif
  template<>
      PyObject* _PyObjFrom(const std::string& c) {
    return PyBytes_FromStringAndSize(c.data(), c.size());
  }

  PyObject* _SwigBytes_FromString(const string& s) {
    return PyBytes_FromStringAndSize(s.data(), s.size());
  }

  // The string must be both ASCII and Unicode compatible, so this routine
  // should be used only for error messages and the like.
  PyObject* _SwigSimpleStr_FromString(const string& s) {
#if PY_MAJOR_VERSION < 3
    return PyString_FromStringAndSize(s.data(), s.size());
#else
    return PyUnicode_FromStringAndSize(s.data(), s.size());
#endif
  }

  template<>
      bool _PyObjAs(PyObject *pyobj, int64 *x) {
#if PY_MAJOR_VERSION < 3
    if (PyInt_Check(pyobj)) {
      *x= static_cast<int64>(PyInt_AsLong(pyobj));
    } else
#endif
    if (PyLong_Check(pyobj)) {
      *x = static_cast<int64>(PyLong_AsLongLong(pyobj));
    } else {
      PyErr_SetString(PyExc_TypeError,
                      "int or long value expected for argument \"$1_name\"");
    }
    return !PyErr_Occurred();
  }
  template<>
      PyObject* _PyObjFrom(const int64 &x) {
    return PyLong_FromLongLong(x);
  }

  template<>
      bool _PyObjAs(PyObject *pyobj, int32 *x) {
#if PY_MAJOR_VERSION < 3
    if (PyInt_Check(pyobj)) {
      *x= static_cast<int32>(PyInt_AsLong(pyobj));
    } else
#endif
    if (PyLong_Check(pyobj)) {
      *x = static_cast<int64>(PyLong_AsLong(pyobj));
    } else {
      PyErr_SetString(PyExc_TypeError,
                      "int or long value expected for argument \"$1_name\"");
    }
    return !PyErr_Occurred();
  }
  template<>
      PyObject* _PyObjFrom(const int32 &x) {
    return PyLong_FromLong(x);
  }

  template <class T>
  bool tf_vector_input_helper(PyObject * seq, std::vector<T> * out,
                              bool (*convert)(PyObject*, T * const)) {
    PyObject *item, *it = PyObject_GetIter(seq);
    if (!it) return false;
    while ((item = PyIter_Next(it))) {
      T elem;
      bool success = convert(item, &elem);
      Py_DECREF(item);
      if (!success) {
        Py_DECREF(it);
        return false;
      }
      if (out) out->push_back(elem);
    }
    Py_DECREF(it);
    return static_cast<bool>(!PyErr_Occurred());
  }

  template <class T>
  static PyObject* tf_vector_output_helper(const std::vector<T>* vec) {
    if (vec == NULL) Py_RETURN_NONE;
    PyObject *lst = PyList_New(vec->size());
    if (lst == NULL) return NULL;
    int i = 0;
    for (const T &x : *vec) {
      PyObject *obj = _PyObjFrom<T>(x);
      if (!obj) {
        Py_DECREF(lst);
        return NULL;
      }
      PyList_SET_ITEM(lst, i++, obj);
    }
    return lst;
  }
%}

%typemap(in) string {
  if (!_PyObjAs<string>($input, &$1)) return NULL;
}

%typemap(in) const string& (string temp) {
  if (!_PyObjAs<string>($input, &temp)) return NULL;
  $1 = &temp;
}

%typemap(out) string {
  $result = PyBytes_FromStringAndSize($1.data(), $1.size());
}

%typemap(out) const string& {
  $result = PyBytes_FromStringAndSize($1->data(), $1->size());
}

%typemap(in, numinputs = 0) string* OUTPUT (string temp) {
  $1 = &temp;
}

%typemap(argout) string * OUTPUT {
  PyObject *str = PyBytes_FromStringAndSize($1->data(), $1->length());
  if (!str) SWIG_fail;
  %append_output(str);
}

%typemap(argout) string* INOUT = string* OUTPUT;

%typemap(varout) string {
  $result = PyBytes_FromStringAndSize($1.data(), $1.size());
}

%typemap(in) int64 {
  if (!_PyObjAs<int64>($input, &$1)) return NULL;
 }

%typemap(in) int32 {
  if (!_PyObjAs<int32>($input, &$1)) return NULL;
}

%typemap(out) int64 {
  $result = PyLong_FromLongLong($1);
}

%typemap(out) int32 {
  $result = PyInt_FromLong($1);
}


%typemap(in) std::vector<tensorflow::fold::tensor_idx_t>(
    std::vector<tensorflow::fold::tensor_idx_t> temp) {
  if (!tf_vector_input_helper(
          $input, &temp, _PyObjAs<tensorflow::fold::tensor_idx_t>)) {
    if (!PyErr_Occurred())
      PyErr_SetString(
          PyExc_TypeError, "sequence(tensorflow::fold::tensor_idx_t) expected");
    return NULL;
  }
  $1 = temp;
}
%typemap(in) const std::vector<tensorflow::fold::tensor_idx_t>& (
    std::vector<tensorflow::fold::tensor_idx_t> temp),
   const std::vector<tensorflow::fold::tensor_idx_t>* (
       std::vector<tensorflow::fold::tensor_idx_t> temp) {
  if (!tf_vector_input_helper(
          $input, &temp, _PyObjAs<tensorflow::fold::tensor_idx_t>)) {
    if (!PyErr_Occurred())
      PyErr_SetString(
          PyExc_TypeError, "sequence(tensorflow::fold::tensor_idx_t) expected");
    return NULL;
  }
  $1 = &temp;
}

%typemap(out) std::vector<tensorflow::fold::tensor_idx_t> {
  $result = tf_vector_output_helper(&$1);
}

%typemap(out) std::vector<tensorflow::fold::tensor_idx_t>& {
  $result = tf_vector_output_helper($1);
}

%define %ignoreall %ignore ""; %enddef
%define %unignore %rename("%s") %enddef
%define %unignoreall %rename("%s") ""; %enddef

%ignoreall

%unignore tensorflow;
%unignore tensorflow::fold;
%unignore tensorflow::fold::Weaver;
%unignore tensorflow::fold::Weaver::Weaver;
%unignore tensorflow::fold::Weaver::~Weaver;
%unignore tensorflow::fold::Weaver::error_string;
%unignore tensorflow::fold::Weaver::Reset;
%unignore tensorflow::fold::Weaver::Serialize;
%unignore tensorflow::fold::Weaver::Deserialize;
%unignore tensorflow::fold::Weaver::MaxDepth;
%unignore tensorflow::fold::Weaver::Deepest;
%unignore tensorflow::fold::Weaver::Depth;
%unignore tensorflow::fold::Weaver::GetTypeShape;
%unignore tensorflow::fold::Weaver::GetNamedTensor;
%unignore tensorflow::fold::Weaver::MakeConstantSerialized;
%unignore tensorflow::fold::Weaver::BatchInput;
%unignore tensorflow::fold::Weaver::CallOp;
%unignore tensorflow::fold::Weaver::AddOutput;
%unignore tensorflow::fold::Weaver::Finalize;
%unignore tensorflow::fold::Weaver::GetWiring;
%unignore tensorflow::fold::Weaver::GetOutputWiring;

%include "tensorflow_fold/loom/weaver.h"

%unignoreall
