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

#ifndef TENSORFLOW_FOLD_LLGTM_PLATFORM_EXTERNAL_H_
#define TENSORFLOW_FOLD_LLGTM_PLATFORM_EXTERNAL_H_

#include <cstdint>
#include <iostream>

#include "tensorflow/core/lib/core/arena.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace llgtm {

using string = std::string;
using int64 = int64_t;
using uint64 = uint64_t;

namespace platform {

// Use arena from TF Core. Use different arena inside Google.
using Arena = tensorflow::core::Arena;

// Tensorflow only allows flags of type float, whereas internal Google
// libraries only allow flags of type double, so we convert as appropriate.
template <class T>
struct FlagType { using type = T; };

template<>
struct FlagType<double> { using type = float; };

// The following macros provide a convenient mechanism for declaring,
// parsing, and using command line flags.  See examples for usage.
// The macros are the public interface; the classes are private.
#define BEGIN_COMMAND_LINE_FLAGS \
  llgtm::platform::CommandLineFlagRegistry global_commandline_flag_registry

#define GET_FLAG_TYPE(TYPE) \
  llgtm::platform::FlagType<TYPE>::type

#define DEFINE_FLAG(TYPE, NAME, DEFVAL, DOCSTR) \
  llgtm::platform::CommandLineFlag<GET_FLAG_TYPE(TYPE)> FLAGS_ ## NAME( \
      #NAME, #TYPE, #DEFVAL, DOCSTR, DEFVAL, &global_commandline_flag_registry)

#define GET_CL_FLAG(NAME) \
  FLAGS_ ## NAME.value()

#define PARSE_COMMAND_LINE_FLAGS(ARGC, ARGV) \
  global_commandline_flag_registry.Parse(&ARGC, ARGV)


namespace { //
class CommandLineFlagRegistry;

// Base class for command line flags.
// Flags will register themselves with a global registry upon construction.
class CommandLineFlagBase {
 public:
  CommandLineFlagBase() = delete;
  CommandLineFlagBase(const CommandLineFlagBase& other) = delete;

  CommandLineFlagBase(const char* name, const char* type,
                      const char* valstr, const char* docstr)
      : name_(name), type_(type), default_value_(valstr), docstring_(docstr)
  {}

  void PrintUsage() {
    std::cout << "--" << name_ << "=" << default_value_
              << "  {" << type_ << "}  " << docstring_ << "\n";
  }

 private:
  friend class CommandLineFlagRegistry;

  const char* name_;
  const char* type_;
  const char* default_value_;
  const char* docstring_;
};


// Derived class for command line flags of a particular type.
template<class T>
class CommandLineFlag : CommandLineFlagBase {
 public:
  CommandLineFlag(const char* name, const char* type,
                  const char* valstr, const char* docstr,
                  T default_value, CommandLineFlagRegistry* registry);

  const T& value() const { return value_; }

 private:
  friend class CommandLineFlagRegistry;

  T value_;
};


// Class which holds a set of command line flags.
// The registry is responsible for parsing command line flags, and for
// printing out usage information.
class CommandLineFlagRegistry {
 public:
  CommandLineFlagRegistry() {}
  ~CommandLineFlagRegistry() {}

  template<class T>
  void Register(CommandLineFlag<T>* flag) {
    flags_.push_back(flag);
    tf_flags_.emplace_back(flag->name_, &flag->value_, flag->docstring_);
  }

  void Parse(int* argc, char** argv) {
    if (*argc > 1) {
      if (string(argv[1]) == string("--help")) {
        PrintUsage();
        exit(0);
      }
    }
    tensorflow::Flags::Parse(argc, argv, tf_flags_);
  }

  void PrintUsage() {
    std::cout << "Usage: " << "\n";
    for (auto* f : flags_) f->PrintUsage();
  }

 private:
  std::vector<CommandLineFlagBase*> flags_;
  std::vector<tensorflow::Flag> tf_flags_;
};


template<class T>
inline CommandLineFlag<T>::CommandLineFlag(const char* name,
                                           const char* type,
                                           const char* valstr,
                                           const char* docstr,
                                           T default_value,
                                           CommandLineFlagRegistry* registry)
    : CommandLineFlagBase(name, type, valstr, docstr), value_(default_value) {
  registry->Register(this);
}
}  // namespace

}  // namespace platform
}  // namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_PLATFORM_EXTERNAL_H_
