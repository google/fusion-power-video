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
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace fpvc {

// Helper functions to convert 16-bit image frames to/from the raw file format.

// Extracts a frame from raw images with 16 bits per pixel, of which 12 bits
// are used.
// The input must have xsize * ysize * 2 bytes, the output xsize * ysize values.
// shift = how much to left shift the value to place the MSB of the 12-bit
// value in the MSB of the 16-bit output.
void ExtractFrame(const uint8_t* frame, size_t xsize,
                  size_t ysize, int shift, bool big_endian,
                  uint16_t* out);

// Converts 16-bit frame back to the original raw file format.
void UnextractFrame(const uint16_t* img,
                    size_t xsize, size_t ysize, int shift,
                    bool big_endian, uint8_t* out);

// Streaming decoder
class StreamingDecoder {
 public:
  /* Decodes frames in a streaming fashing. Appends the given bytes to the
  input buffer. Calls the callback function for all decoded frames that could
  be decoded so far. The payload is an optional parameter to pass on to the
  callback. */
  void Decode(const uint8_t* bytes, size_t size,
      std::function<void(bool ok, uint16_t* frame, size_t xsize, size_t ysize,
          void* payload)> callback,
      void* payload = nullptr);

 private:
  size_t xsize;
  size_t ysize;

  size_t id = 0;

  std::vector<uint16_t> delta_frame;

  std::vector<uint8_t> buffer;
};

// Rnadom access decoder: requires random access to the entire data file,
// can decode any frame in any order.
class RandomAccessDecoder {
 public:
   // Parses the header and footer, must be called once with the full data
   // before using DecodeFrame, xsize, ysize or numframes.
   bool Init(const uint8_t* data, size_t size);

   // Decodes the frame with the given index. The index must be smaller than
   // numframes. The output frame must have xsize * ysize values.
   bool DecodeFrame(size_t index, uint16_t* frame) const;

   size_t xsize() const { return xsize_; }
   size_t ysize() const { return ysize_; }

   // Returns amount of frames in the full file.
   size_t numframes() const { return frame_offsets.size(); }

 private:
  size_t xsize_ = 0;
  size_t ysize_ = 0;
  std::vector<uint16_t> delta_frame;
  std::vector<size_t> frame_offsets;
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

// Multithreaded encoder.
class Encoder {
 public:
  // Uses num_threads worker threads, or disables multithreading if num_threads
  // is 0.
  Encoder(size_t num_threads = 4);

  // The payload is an optional argument to pass from calls to the callback.
  typedef std::function<void(const uint8_t* compressed, size_t size,
      void* payload)> Callback;

  /* Initializes before the first frame, and writes the header bytes by
  outputting them to the callback.
  The delta_frame must have xsize * ysize pixels. */
  void Init(const uint16_t* delta_frame, size_t xsize, size_t ysize,
      Callback callback, void* payload);

  /* Encodes a single 16-bit grayscale frame, should be extracted from the raw
  data using using ExtractFrame. Calls the callback function when finished
  compressing, asynchronously but guarded and guaranteed in the correct order.
  Payload can optionally be used to bind an extra argument to pass to the
  callback. User must manage memory of img and prev, both must exist until the
  callback for this frame is called.
  Init must be called before compressing the first frame, and Finish must be
  called after the last frame.*/
  void CompressFrame(const uint16_t* img, Callback callback,
      void* payload);

  /* Waits and finishes all threads, and writes the footer bytes by
  outputting them to the callback. */
  void Finish(Callback callback, void* payload);

 private:
  struct Task {
    const uint16_t* frame;
    size_t id;
    Callback callback;
    void* payload;
  };

  void RunThread();

  std::vector<uint8_t> RunTask(const Task& task);

  // Finalize a task, unlike RunTask this is guaranteed to run in sequential
  // order and guarded.
  void FinishTask(const Task& task, std::vector<uint8_t>* compressed);

  void WriteFrameIndex(std::vector<uint8_t>* compressed) const;

  std::vector<std::thread*> threads;
  std::mutex m;

  std::condition_variable cv_in;  // for incoming frames
  std::queue<Task> q_in;

  std::condition_variable cv_out;  // for outputting compressed frames
  std::queue<Task> q_out;

  std::condition_variable cv_main;  // for the main thread

  bool finish = false;

  size_t id = 0;  // Unique frame id.

  size_t xsize_;
  size_t ysize_;

  std::vector<uint16_t> delta_frame_;
  std::vector<size_t> frame_offsets;
  size_t bytes_written = 0;
};

}  // namespace fpvc

#endif  // FUSION_POWER_VIDEO_H_
