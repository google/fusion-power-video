// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FUSION_POWER_VIDEO_H_
#define FUSION_POWER_VIDEO_H_

#include <stdint.h>

#include <vector>

std::vector<uint16_t> ExtractFrame(const uint8_t* frame, size_t sizex,
                                   size_t sizey, int shift, bool big_endian);

std::vector<uint8_t> CompressFrame(std::vector<uint16_t> img,
                                   const std::vector<uint16_t>& prev,
                                   size_t sizex, size_t sizey);

std::vector<uint16_t> DecompressFrame(const std::vector<uint16_t>& prev,
                                      const uint8_t* in, size_t size,
                                      size_t* pos);

std::vector<uint8_t> UnextractFrame(const std::vector<uint16_t>& img,
                                    size_t sizex, size_t sizey, int shift,
                                    bool big_endian);

uint64_t ReadVarint(const uint8_t* data, size_t size, size_t* pos);

#endif  // FUSION_POWER_VIDEO_H_
