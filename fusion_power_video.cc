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

#include "fusion_power_video.h"

#include <brotli/decode.h>
#include <brotli/encode.h>
#include <math.h>

#include <functional>
#include <iostream>
#include <thread>

namespace fpvc {
namespace {

bool BrotliCompress(const uint8_t* in, size_t size, std::vector<uint8_t>* out,
                    int quality = 11, int windowbits = BROTLI_MAX_WINDOW_BITS) {
  // A safe size is BrotliEncoderMaxCompressedSize(size) + 1024, but making the
  // buffer smaller and only increasing in the rare case it's needed is faster.
  size_t out_size = size;
  out->resize(out_size);
  std::unique_ptr<BrotliEncoderState, std::function<void(BrotliEncoderState*)>>
      enc(BrotliEncoderCreateInstance(nullptr, nullptr, nullptr),
          &BrotliEncoderDestroyInstance);
  if (!enc) return false;
  BrotliEncoderSetParameter(enc.get(), BROTLI_PARAM_QUALITY, quality);
  BrotliEncoderSetParameter(enc.get(), BROTLI_PARAM_LGWIN, windowbits);
  BrotliEncoderSetParameter(enc.get(), BROTLI_PARAM_SIZE_HINT, size);
  size_t avail_in = size;
  const uint8_t* next_in = in;
  size_t avail_out = out_size;
  uint8_t* next_out = out->data();
  do {
    if (avail_out == 0) {
      size_t pos = next_out - out->data();
      size_t more = out_size >> 1u;
      out_size += more;
      out->resize(out_size);
      avail_out += more;
      next_out = out->data() + pos;
    }
    if (!BrotliEncoderCompressStream(enc.get(), BROTLI_OPERATION_FINISH,
                                     &avail_in, &next_in, &avail_out, &next_out,
                                     nullptr)) {
      return false;
    }
  } while (!BrotliEncoderIsFinished(enc.get()));
  out_size -= avail_out;
  out->resize(out_size);
  return true;
}

// pos = where to start, and outputs position of end of stream, allowing to then
// continue if there are more concatenated streams (or know where the valid
// brotli stream ended)
bool BrotliDecompress(const uint8_t* in, size_t size, size_t* pos,
                      std::vector<uint8_t>* out) {
  std::unique_ptr<BrotliDecoderState, std::function<void(BrotliDecoderState*)>>
      decoder(BrotliDecoderCreateInstance(nullptr, nullptr, nullptr),
              &BrotliDecoderDestroyInstance);
  if (!decoder) return false;

  size_t avail_in = size - *pos;
  const uint8_t* next_in = in + *pos;
  BrotliDecoderResult result;
  for (;;) {
    size_t avail_out = 0;
    result = BrotliDecoderDecompressStream(decoder.get(), &avail_in, &next_in,
                                           &avail_out, nullptr, nullptr);
    if (result != BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT) {
      break;
    }
    size_t out_size = 0;
    const uint8_t* out_buf = BrotliDecoderTakeOutput(decoder.get(), &out_size);
    if (out_size > 0) {
      out->insert(out->end(), out_buf, out_buf + out_size);
    }
  }
  *pos = size - avail_in;
  return result == BROTLI_DECODER_RESULT_SUCCESS;
}

void WriteVarint(uint64_t value, std::vector<uint8_t>* out) {
  for (;;) {
    out->push_back((value & 127u) | (value > 127u ? 128u : 0u));
    value >>= 7u;
    if (!value) return;
  }
}

size_t VarintSize(uint64_t value) {
  std::vector<uint8_t> temp;
  WriteVarint(value, &temp);
  return temp.size();
}

// Returns average entropy per symbol (a guess of bits per pixel).
float EstimateEntropy(const std::vector<size_t>& v) {
  float result = 0;
  float sum = 0;
  for (size_t i = 0; i < v.size(); i++) {
    sum += v[i];
  }
  if (!sum) return 0;
  float s = 1.0 / sum;
  for (size_t i = 0; i < v.size(); i++) {
    if (!v[i]) continue;
    float p = v[i] * s;
    // TODO: use a fast log2 approximation
    result -= log2(p) * p;
  }
  return result;
}

// clamped gradient predictor
uint16_t ClampedGradient(uint16_t n, uint16_t w, uint16_t nw) {
  const uint16_t i = std::min(n, w), a = std::max(n, w);
  const uint16_t gradient = n + w - nw;
  const uint16_t clamped = (nw < i) ? a : gradient;
  return (nw > a) ? i : clamped;
}

}  // namespace



uint64_t ReadVarint(const uint8_t* data, size_t size, size_t* pos) {
  uint64_t result = 0;
  size_t num = 0;
  for (;;) {
    if (*pos >= size || num > 9) return 0;
    uint8_t byte = data[(*pos)++];
    uint64_t shift = (num++) * 7;
    result |= (uint64_t)(byte & 127) << shift;
    if (byte < 128) return result;
  }
}

// Extracts a frame from raw images with 16 bits per pixel, of which 12 bits
// are used.
// shift = how much to left shift the value to place the MSB of the 12-bit
// value in the MSB of the 16-bit output.
std::vector<uint16_t> ExtractFrame(const uint8_t* frame, size_t xsize,
                                   size_t ysize, int shift, bool big_endian) {
  std::vector<uint16_t> img(xsize * ysize);
  for (size_t i = 0; i < img.size(); i++) {
    uint8_t high = frame[i * 2 + (big_endian ? 0 : 1)];
    uint8_t low = frame[i * 2 + (big_endian ? 1 : 0)];
    img[i] = (high << 8u) | low;
    img[i] <<= shift;
  }
  return img;
}

std::vector<uint8_t> UnextractFrame(const std::vector<uint16_t>& img,
                                    size_t xsize, size_t ysize, int shift,
                                    bool big_endian) {
  std::vector<uint8_t> data(xsize * ysize * 2);
  for (size_t i = 0; i < img.size(); i++) {
    uint16_t u = img[i];
    u >>= shift;
    uint8_t a = u & 255;
    uint8_t b = u >> 8;
    if (big_endian) std::swap(a, b);
    data[i * 2 + 0] = a;
    data[i * 2 + 1] = b;
  }
  return data;
}

std::vector<uint8_t> CompressFrame(std::vector<uint16_t> img,
                                   const std::vector<uint16_t>& prev,
                                   size_t xsize, size_t ysize) {
  int use_delta = 0;
  int use_vertical = 0;
  int use_horizontal = 0;
  int use_pred = 0;

#define ENABLE_PREDICTORS 1
#define USE_CLAMPED_GRADIENT 0  // If not, uses hor or ver instead

#if ENABLE_PREDICTORS
  if (!prev.empty()) {
    // heuristic to choose to use delta
    size_t skip = 15;  // let the heuristic run faster
    std::vector<size_t> counta0(256);
    std::vector<size_t> counta1(256);
    std::vector<size_t> countd0(256);
    std::vector<size_t> countd1(256);

    for (size_t i = 0; i < img.size(); i += skip) {
      uint16_t a = img[i];
      uint16_t d = a - prev[i];
      counta0[a & 255]++;
      counta1[a >> 8]++;
      countd0[d & 255]++;
      countd1[d >> 8]++;
    }
    float ea = EstimateEntropy(counta0) + EstimateEntropy(counta1);
    float ed = EstimateEntropy(countd0) + EstimateEntropy(countd1);

    use_delta = ed < ea;
    if (use_delta) {
      for (size_t i = 0; i < img.size(); i++) {
        img[i] = img[i] - prev[i] + 32768u;
      }
    }
  }

  // TODO: try a simple context model

#if USE_CLAMPED_GRADIENT
  {
    std::vector<size_t> counta0(256);
    std::vector<size_t> counta1(256);
    std::vector<size_t> countb0(256);
    std::vector<size_t> countb1(256);

    size_t skip = 31;  // let the heuristic run faster
    for (size_t i = xsize + 1; i < img.size(); i += skip) {
      uint16_t a = img[i];
      uint16_t n = img[i - xsize];
      uint16_t w = img[i - 1];
      uint16_t nw = img[i - xsize - 1];
      uint16_t b = a - ClampedGradient(n, w, nw);
      counta0[a & 255]++;
      counta1[(a >> 8) & 255]++;
      countb0[b & 255]++;
      countb1[(b >> 8) & 255]++;
    }

    float ea = EstimateEntropy(counta0) + EstimateEntropy(counta1);
    float eb = EstimateEntropy(countb0) + EstimateEntropy(countb1);

    if (eb < ea) use_pred = 1;

    if (use_pred) {
      std::vector<uint16_t> temp = img;
      for (size_t i = xsize + 1; i < img.size(); i++) {
        uint16_t n = img[i - xsize];
        uint16_t w = img[i - 1];
        uint16_t nw = img[i - xsize - 1];
        temp[i] = img[i] - ClampedGradient(n, w, nw);
      }
      temp.swap(img);
    }
  }
#else  // USE_CLAMPED_GRADIENT
  {
    std::vector<size_t> counta0(256);
    std::vector<size_t> counth0(256);
    std::vector<size_t> countv0(256);
    std::vector<size_t> counta1(256);
    std::vector<size_t> counth1(256);
    std::vector<size_t> countv1(256);

    uint64_t sumh = 0, sumv = 0;
    size_t skip = 15;  // let the heuristic run faster
    for (size_t i = xsize; i < img.size(); i += skip) {
      uint16_t a = img[i];
      uint16_t h = img[i - 1];
      uint16_t v = img[i - xsize];
      h = (a - h);
      v = (a - v);
      counta0[(a)&255]++;
      counth0[(h)&255]++;
      countv0[(v)&255]++;
      counta1[(a >> 8) & 255]++;
      counth1[(h >> 8) & 255]++;
      countv1[(v >> 8) & 255]++;
    }

    float ea = EstimateEntropy(counta0) + EstimateEntropy(counta1);
    float eh = EstimateEntropy(counth0) + EstimateEntropy(counth1);
    float ev = EstimateEntropy(countv0) + EstimateEntropy(countv1);

    if (eh < ea && eh < ev) {
      use_horizontal = 1;
    } else if (ev < ea) {
      use_vertical = 1;
    }

    if (use_vertical) {
      std::vector<uint16_t> temp = img;
      for (size_t i = xsize; i < img.size(); i++) {
        temp[i] = img[i] - img[i - xsize];
      }
      temp.swap(img);
    }
    if (use_horizontal) {
      std::vector<uint16_t> temp = img;
      use_horizontal = 1;
      for (size_t i = 1; i < img.size(); i++) {
        temp[i] = img[i] - img[i - 1];
      }
      temp.swap(img);
    }
  }
#endif  // USE_CLAMPED_GRADIENT
#endif  // ENABLE_PREDICTORS

  std::vector<uint8_t> low;
  std::vector<uint8_t> lowc;
  std::vector<uint8_t> high;
  std::vector<uint8_t> highc;

  low.resize(img.size());
  high.resize(img.size());
  for (size_t i = 0; i < img.size(); i++) {
    low[i] = img[i] & 255;
    high[i] = (img[i] >> 8u) & 255;
  }

  // NOTE: for this use case, brotli quality 1 gives smaller result than
  // brotli quality 2, yet is faster. Only the entropy coding matters, not the
  // LZ77.
  int quality = 1;
  BrotliCompress(low.data(), low.size(), &lowc, quality);
  BrotliCompress(high.data(), high.size(), &highc, quality);

  std::vector<uint8_t> result;
  size_t result_size =
      VarintSize(xsize) + VarintSize(ysize) + 1 + highc.size() + lowc.size();
  WriteVarint(result_size, &result);
  result.push_back(use_delta + (use_vertical << 1) + (use_horizontal << 2) +
                   (use_pred << 3));
  WriteVarint(xsize, &result);
  WriteVarint(ysize, &result);
  result.insert(result.end(), lowc.begin(), lowc.end());
  result.insert(result.end(), highc.begin(), highc.end());

  return result;
}

std::vector<uint16_t> DecompressFrame(const std::vector<uint16_t>& prev,
                                      const uint8_t* in, size_t size,
                                      size_t* pos) {
  if (*pos >= size) return {};

  size_t expected_size = ReadVarint(in, size, pos);
  if (*pos + expected_size > size) return {};

  if (*pos >= size) return {};
  bool use_delta = in[*pos] & 1;
  bool use_vertical = in[*pos] & 2;
  bool use_horizontal = in[*pos] & 4;
  bool use_pred = in[*pos] & 8;
  (*pos)++;

  size_t xsize = ReadVarint(in, size, pos);
  size_t ysize = ReadVarint(in, size, pos);
  if (*pos >= size) return {};

  // Error: invalid dimensions or varints
  if (!xsize || !ysize) return {};

  // Error: want to use inter-frame delta but previous frame not supplied.
  if (use_delta && prev.empty()) return {};

  std::vector<uint8_t> low;
  if (!BrotliDecompress(in, size, pos, &low)) return {};

  std::vector<uint8_t> high;
  if (!BrotliDecompress(in, size, pos, &high)) return {};

  // Error: sizes don't match image size
  if (low.size() != xsize * ysize) return {};
  if (low.size() != xsize * ysize) return {};
  if (use_delta && prev.size() != xsize * ysize) return {};

  std::vector<uint16_t> img(low.size());
  for (size_t i = 0; i < img.size(); i++) {
    img[i] = (high[i] << 8) | low[i];
  }

  if (use_pred) {
    for (size_t i = xsize + 1; i < img.size(); i++) {
      uint16_t n = img[i - xsize];
      uint16_t w = img[i - 1];
      uint16_t nw = img[i - xsize - 1];
      img[i] = img[i] + ClampedGradient(n, w, nw);
    }
  }

  if (use_horizontal) {
    for (size_t i = 1; i < img.size(); i++) {
      img[i] = img[i] + img[i - 1];
    }
  }

  if (use_vertical) {
    for (size_t i = xsize; i < img.size(); i++) {
      img[i] = img[i] + img[i - xsize];
    }
  }

  if (use_delta) {
    for (size_t i = 0; i < img.size(); i++) {
      img[i] = img[i] + prev[i] - 32768u;
    }
  }

  return img;
}

Encoder::Encoder(size_t xsize, size_t ysize, size_t num_threads) :
    xsize(xsize), ysize(ysize) {
  threads.resize(num_threads);
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i] = new std::thread(&Encoder::RunThread, this);
  }
}

Encoder::~Encoder() {
  Join();
}

void Encoder::Join() {
  {
    std::unique_lock<std::mutex> l(m);
    if (finish) return;  // Already done.
    cv_main.wait(l, [this]{return q_out.empty();});
    finish = true;
  }
  cv_in.notify_all();

  for (size_t i = 0; i < threads.size(); i++) {
    threads[i]->join();
    delete threads[i];
  }
}

void Encoder::CompressFrame(const std::vector<uint16_t>* img,
    const std::vector<uint16_t>* prev,
    void (*Callback)(const uint8_t* compressed, size_t size, void* payload),
    void* payload) {
  Task task;
  task.frame = img;
  task.prev = prev;
  task.id = id++;
  task.Callback = Callback;
  task.payload = payload;

  {
    std::unique_lock<std::mutex> l(m);
    q_in.push(task);
    q_out.push(task);
  }
  cv_in.notify_one();

  {
    std::unique_lock<std::mutex> l(m);
    // Wait if the queue gets too full to prevent out of memory.
    cv_main.wait(l, [this]{return q_out.size() < threads.size() * 4;});
  }
}

void Encoder::RunThread() {
  for (;;) {
    Task task;

    // Wait for starting a new compression
    {
      std::unique_lock<std::mutex> l(m);
      cv_in.wait(l, [this]{return !q_in.empty() || finish;});
      if (finish) return;
      task = q_in.front();
      q_in.pop();
    }

    auto compressed =
        fpvc::CompressFrame(*task.frame, *task.prev, xsize, ysize);
    // Wait for outputting in the correct order.
    {
      std::unique_lock<std::mutex> l(m);
      cv_out.wait(l, [&task, this]{return q_out.front().id == task.id;});
      q_out.pop();
      task.Callback(compressed.data(), compressed.size(), task.payload);
    }
    // Finished outputting
    cv_out.notify_all();
    // Changed q_out size
    cv_main.notify_one();
  }
}

}  // namespace fpvc

