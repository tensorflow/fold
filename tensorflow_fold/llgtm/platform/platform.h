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

#ifndef TENSORFLOW_FOLD_LLGTM_PLATFORM_PLATFORM_H_
#define TENSORFLOW_FOLD_LLGTM_PLATFORM_PLATFORM_H_

// A platform defines aliases for classes such as Arena and includes headers
// for "base" functionality such as logging.

// There are two platforms: GOOGLE and EXTERNAL (open sourced version).
// EXTERNAL falls back to TF as a dependency, whereas GOOGLE can use internal
// headers and dependencies.

#if defined(LLGTM_PLATFORM_GOOGLE) && defined(LLGTM_PLATFORM_EXTERNAL)
#error "Multiple platforms selected. Include one platform in BUILD target."
#endif

#if defined(LLGTM_PLATFORM_GOOGLE)
#include "tensorflow_fold/llgtm/platform/google.h"
#elif defined(LLGTM_PLATFORM_EXTERNAL)
#include "tensorflow_fold/llgtm/platform/external.h"
#else
#error "No platform selected. Include one platform in BUILD target."
#endif

#endif  // TENSORFLOW_FOLD_LLGTM_PLATFORM_PLATFORM_H_
