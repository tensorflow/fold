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

#ifndef TENSORFLOW_FOLD_LLGTM_DEVICE_H_
#define TENSORFLOW_FOLD_LLGTM_DEVICE_H_

#include "absl/strings/string_view.h"

namespace llgtm {

// Identifies a device such as GPU or CPU.
using DeviceID = uint8_t;

// TODO(matthiasspringer): Replace hard-coded device IDs with proper device
// management.
enum Devices : DeviceID {
  kDeviceIDCPU = 0,
  kDeviceIDGPU = 1,

  // GPU device is available only in CUDA builds. Maximum ID is an
  // exclusive bound.
#ifdef GOOGLE_CUDA
  kDeviceMaximumID = 2,
#else
  kDeviceMaximumID = 1,
#endif

  kDeviceIDUnspecified = 255
};

// Return the name of the given device.
absl::string_view device_name(DeviceID device);

}  // namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_DEVICE_H_
