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

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace fpvc {

std::vector<uint16_t> ExtractFrame(const uint8_t* frame, size_t sizex,
                                   size_t sizey, int shift, bool big_endian);

// Compresses a frame single-threaded. Use Encoder for multithreaded compression
// instead.
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

// Multithreaded encoder for CompressFrame.
class Encoder {
 public:
  Encoder(size_t xsize, size_t ysize, size_t num_threads = 4);

  // Calls Join if it was not yet called.
  virtual ~Encoder();

  /* Encodes a single 16-bit grayscale frame, should be extracted from the raw
  data using using ExtractFrame. Calls the Callback function when finished
  compressing, asynchronously but guarded and guaranteed in the correct order.
  Payload can optionally be used to bind an extra argument to pass to the
  callback. User must manage memory of img and prev, both must exist until the
  callback for this frame is called. */
  void CompressFrame(const std::vector<uint16_t>* img,
      const std::vector<uint16_t>* prev,
      void (*Callback)(const uint8_t* compressed, size_t size, void* payload),
      void* payload = nullptr);

  // Waits and finishes all threads.
  void Join();

 private:
  struct Task {
    const std::vector<uint16_t>* frame;
    const std::vector<uint16_t>* prev;
    size_t id;
    void* payload;
    void (*Callback)(const uint8_t* compressed, size_t size, void* payload);
  };

  void RunThread();

  std::vector<std::thread*> threads;
  std::mutex m;

  std::condition_variable cv_in;  // for incoming frames
  std::queue<Task> q_in;

  std::condition_variable cv_out;  // for outputting compressed frames
  std::queue<Task> q_out;

  std::condition_variable cv_main;  // for the main thread

  bool finish = false;

  size_t id = 0;

  size_t xsize;
  size_t ysize;
};

}  // namespace fpvc

#endif  // FUSION_POWER_VIDEO_H_
