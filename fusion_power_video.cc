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

#include <math.h>
#include <string.h>  // memcpy

#include <functional>
#include <iostream>
#include <thread>

#include <brotli/decode.h>
#include <brotli/encode.h>

/*
Description of the file format:

A fusion power video file contains 1 or more grayscale 16-bit image frames, and
uses the format described below. To decode a frame, only the main header, the
one static delta frame and compressed data for the frame itself are needed,
frames do not depend on other frames. Each format section below describes the
concatenation of one or more streams or values encoded in bytes (octets),
possibly referring to sub-sections to further define a stream, culminating in
the full file format.

full file format:
-header: see header format below
-encoded delta frame: see frame format below
-one or more frames: per frame:
--encoded frame: see frame format below
-footer: see footer format (frame index) below
Note: the encoder can only choose a single delta frame that can be used for
prediction of all the frames (so frames don't depend on each other, only on
the one delta frame); the encoder should make a good choice of delta frame
for good compression (example: the first frame, assuming the camera remains
static and the first frame is a full representative image).

header format:
-4 bytes: xsize (little endian 32-bit integer)
-4 bytes: ysize (little endian 32-bit integer)

frame format:
-4 bytes: compressed size of entire frame, including these 4 bytes (little
 endian 32-bit integer)
-1 byte: flags, see below
-variable amount of bytes: brotli compressed low bytes
-variable amount of bytes: brotli compressed high bytes
Note: the brotli decoder knows where the first brotli stream ends so the
split point is known during decoding. The brotli format is specified in
RFC 7932. See below for the complete procedure to decode a frame.

footer format (frame index):
-4 bytes: size of this entire footer, including these 4 bytes (little
 endian 32-bit integer)
-1 byte: flags, must have value 2
-per frame:
--8 bytes: offset from the start of the file to the start of this frame (little
  endian 64-bit integer)
-8 bytes: amount of frames
Note: if the full file is available, the decoder can compute the start of the
footer by parsing the last 8 bytes, rather than jumping frame by frame through
the entire file from the front, in case one wants to decode only a particular
frame.

flags meanings:
-flags & 1: this must be true for the delta frame immediately after the header,
 and false for all other frames. Indicates this is not a frame to be decoded,
 but the delta frame that all other frames can use as base for prediction.
-flags & 2: this must be true for the footer (frame index), and must be false
 for all frames.
-flags & 4: iff true, delta frame prediction is enabled. This must be false if
 this frame is the delta frame.
-flags & 8: iff true, vertical prediction is enabled
-flags & 16: iff true, horizontal prediction is enabled

procedure to decode a frame:
-given the xsize and ysize from the header, a frame has xsize columns and
 ysize rows.
-brotli decompress the low bytes. These correspond to the LSB's of the 16-bit
 image.
-brotli decompress the high bytes. These correspond to the MSB's of the 16-bit
 image.
-Note: the brotli format is specified in RFC 7932
-Note: the format has two concatenated brotli streams in a row, their individual
 sizes are not encoded in this file format, but they are implicitely present
 in the brotli stream, and the brotli decoder knows where the first stream ends.
 The second stream must end at the last byte of this encoded frame, if not the
 file is invalid.
-Note: each brotli-decoded byte stream has xsize * ysize bytes, if not the file
 is invalid.
-combine the low and high byte streams into a single xsize*ysize 16-bit frame of
 unsigned 16-bit integers.
-if horizontal prediction is enabled, then for all pixels except those of the
 leftmost column, compute: new_value = old_value + new_value_left, with
 new_value_left the new value of the pixel left of this one. The addition
 should wrap on overflow.
-if vertical prediction is enabled, then for all pixels except those of the
 topmost row, compute: new_value = old_value + new_value_up, similar to the
 horizontal prediction but with up refering to the pixel above the current
 pixel.
-if delta prediction is enabled, then for all pixels compute:
 new_value = old_value + delta_frame_value - 32768, similar to the
 horizontal prediction but with delta_frame_value the value at the corresponding
 position from the delta frame.
-the resulting 16-bit image may represent 12-bit data (or another bit amount),
 in that case, the least significant bits will be set to 0.
*/

namespace fpvc {
namespace {

// Enable this to print line number and message on decoding errors.
//#define FAIL_DEBUG_MESSAGE

#ifdef FAIL_DEBUG_MESSAGE
bool FailPrint(const char* file, int line, const std::string& message = "") {
  std::cerr << "failure at: " << file << ":" << line;
  if (!message.empty()) std::cerr << ": " << message;
  std::cerr << std::endl;
  return false;
}
#define FAILURE(message) FailPrint(__FILE__, __LINE__, std::string(message))
#else  // FAIL_DEBUG_MESSAGE
#define FAILURE(message) false
#endif  // FAIL_DEBUG_MESSAGE

bool BrotliCompress(const uint8_t* in, size_t size, std::vector<uint8_t>* out,
                    int quality = 11, int windowbits = BROTLI_MAX_WINDOW_BITS) {
  // A safe size is BrotliEncoderMaxCompressedSize(size) + 1024, but making the
  // buffer smaller and only increasing in the rare case it's needed is faster.
  size_t out_size = size;
  out->resize(out_size);
  std::unique_ptr<BrotliEncoderState, std::function<void(BrotliEncoderState*)>>
      enc(BrotliEncoderCreateInstance(nullptr, nullptr, nullptr),
          &BrotliEncoderDestroyInstance);
  if (!enc) return FAILURE();
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
      return FAILURE();
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
  if (!decoder) return FAILURE();

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
  if (result != BROTLI_DECODER_RESULT_SUCCESS) return FAILURE();
  return true;
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

uint32_t ReadUint32LE(const uint8_t* data) {
  return (uint32_t)data[0] + ((uint32_t)data[1] << 8) +
      ((uint32_t)data[2] << 16) + ((uint32_t)data[3] << 24);
}

void WriteUint32LE(uint32_t value, uint8_t* data) {
  data[0] = value & 255;
  data[1] = (value >> 8) & 255;
  data[2] = (value >> 16) & 255;
  data[3] = (value >> 24) & 255;
}

uint64_t ReadUint64LE(const uint8_t* data) {
  return (uint64_t)data[0] + ((uint64_t)data[1] << 8) +
      ((uint64_t)data[2] << 16) + ((uint64_t)data[3] << 24) +
      ((uint64_t)data[4] << 32) + ((uint64_t)data[5] << 40) +
      ((uint64_t)data[6] << 48) + ((uint64_t)data[7] << 56);
}

void WriteUint64LE(uint64_t value, uint8_t* data) {
  data[0] = value & 255;
  data[1] = (value >> 8) & 255;
  data[2] = (value >> 16) & 255;
  data[3] = (value >> 24) & 255;
  data[4] = (value >> 32) & 255;
  data[5] = (value >> 40) & 255;
  data[6] = (value >> 48) & 255;
  data[7] = (value >> 56) & 255;
}

void CompressFrame(const uint16_t* delta_frame,
                   const uint16_t* img,
                   size_t xsize, size_t ysize,
                   std::vector<uint8_t>* out) {
  int use_delta = 0;
  int use_vertical = 0;
  int use_horizontal = 0;
  int use_pred = 0;
  size_t numpixels = xsize * ysize;
  std::vector<uint16_t> img_copy(img, img + numpixels);

#define ENABLE_PREDICTORS 1
#define USE_CLAMPED_GRADIENT 0  // If not, uses hor or ver instead

#if ENABLE_PREDICTORS
  if (delta_frame) {
    // heuristic to choose to use delta
    size_t skip = 15;  // let the heuristic run faster
    std::vector<size_t> counta0(256);
    std::vector<size_t> counta1(256);
    std::vector<size_t> countd0(256);
    std::vector<size_t> countd1(256);

    for (size_t i = 0; i < numpixels; i += skip) {
      uint16_t a = img_copy[i];
      uint16_t d = a - delta_frame[i];
      counta0[a & 255]++;
      counta1[a >> 8]++;
      countd0[d & 255]++;
      countd1[d >> 8]++;
    }
    float ea = EstimateEntropy(counta0) + EstimateEntropy(counta1);
    float ed = EstimateEntropy(countd0) + EstimateEntropy(countd1);

    use_delta = ed < ea;
    if (use_delta) {
      for (size_t i = 0; i < numpixels; i++) {
        img_copy[i] = img_copy[i] - delta_frame[i] + 32768u;
      }
    }
  }


#if USE_CLAMPED_GRADIENT
  {
    std::vector<size_t> counta0(256);
    std::vector<size_t> counta1(256);
    std::vector<size_t> countb0(256);
    std::vector<size_t> countb1(256);

    size_t skip = 31;  // let the heuristic run faster
    for (size_t i = xsize + 1; i < numpixels; i += skip) {
      uint16_t a = img_copy[i];
      uint16_t n = img_copy[i - xsize];
      uint16_t w = img_copy[i - 1];
      uint16_t nw = img_copy[i - xsize - 1];
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
      std::vector<uint16_t> temp(numpixels);
      for (size_t i = xsize + 1; i < img_copy.size(); i++) {
        uint16_t n = img_copy[i - xsize];
        uint16_t w = img_copy[i - 1];
        uint16_t nw = img_copy[i - xsize - 1];
        temp[i] = img_copy[i] - ClampedGradient(n, w, nw);
      }
      img_copy.swap(temp);
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

    size_t skip = 15;  // let the heuristic run faster
    for (size_t i = xsize; i < img_copy.size(); i += skip) {
      uint16_t a = img_copy[i];
      uint16_t h = img_copy[i - 1];
      uint16_t v = img_copy[i - xsize];
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
      std::vector<uint16_t> temp = img_copy;
      for (size_t i = xsize; i < img_copy.size(); i++) {
        temp[i] = img_copy[i] - img_copy[i - xsize];
      }
      temp.swap(img_copy);
    }
    if (use_horizontal) {
      std::vector<uint16_t> temp = img_copy;
      use_horizontal = 1;
      for (size_t i = 1; i < img_copy.size(); i++) {
        temp[i] = img_copy[i] - img_copy[i - 1];
      }
      temp.swap(img_copy);
    }
  }
#endif  // USE_CLAMPED_GRADIENT
#endif  // ENABLE_PREDICTORS

  std::vector<uint8_t> low;
  std::vector<uint8_t> lowc;
  std::vector<uint8_t> high;
  std::vector<uint8_t> highc;

  low.resize(img_copy.size());
  high.resize(img_copy.size());
  for (size_t i = 0; i < img_copy.size(); i++) {
    low[i] = img_copy[i] & 255;
    high[i] = (img_copy[i] >> 8u) & 255;
  }

  // NOTE: for this use case, brotli quality 1 gives smaller result than
  // brotli quality 2, yet is faster. Only the entropy coding matters, not the
  // LZ77.
  int quality = 1;
  BrotliCompress(low.data(), low.size(), &lowc, quality);
  BrotliCompress(high.data(), high.size(), &highc, quality);

  size_t result_size = 4 + 1 + highc.size() + lowc.size();
  out->resize(out->size() + result_size);
  uint8_t* out_data = out->data() + out->size() - result_size;
  WriteUint32LE(result_size, out_data);
  uint8_t flags = (use_delta << 2) + (use_vertical << 3) +
                  (use_horizontal << 4) + (use_pred << 5);
  out_data[4] = flags;
  memcpy(out_data + 5, lowc.data(), lowc.size());
  memcpy(out_data + 5 + lowc.size(), highc.data(), highc.size());
}

bool DecompressFrame(const uint16_t* delta_frame,
                     const uint8_t* in, size_t size,
                     size_t xsize, size_t ysize,
                     size_t* pos, uint16_t* img) {
  if (*pos + 5 > size) return FAILURE();
  size_t expected_size = ReadUint32LE(in + *pos);
  if (*pos + expected_size > size) return FAILURE();

  uint8_t flags = in[*pos + 4];
  bool use_delta = flags & 4;
  bool use_vertical = flags & 8;
  bool use_horizontal = flags & 16;
  bool use_pred = flags & 32;
  *pos += 5;
  // Error: invalid dimensions
  if (!xsize || !ysize) return FAILURE();
  size_t numpixels = xsize * ysize;
  // Error: want to use inter-frame delta but delta_frame frame not supplied.
  if (use_delta && !delta_frame) return FAILURE();

  std::vector<uint8_t> low;
  if (!BrotliDecompress(in, size, pos, &low)) return FAILURE();

  std::vector<uint8_t> high;
  if (!BrotliDecompress(in, size, pos, &high)) return FAILURE();

  // Error: sizes don't match image size
  if (low.size() != numpixels) return FAILURE();
  if (high.size() != numpixels) return FAILURE();

  for (size_t i = 0; i < numpixels; i++) {
    img[i] = (high[i] << 8) | low[i];
  }

  if (use_pred) {
    for (size_t i = xsize + 1; i < numpixels; i++) {
      uint16_t n = img[i - xsize];
      uint16_t w = img[i - 1];
      uint16_t nw = img[i - xsize - 1];
      img[i] = img[i] + ClampedGradient(n, w, nw);
    }
  }

  if (use_horizontal) {
    for (size_t i = 1; i < numpixels; i++) {
      img[i] = img[i] + img[i - 1];
    }
  }

  if (use_vertical) {
    for (size_t i = xsize; i < numpixels; i++) {
      img[i] = img[i] + img[i - xsize];
    }
  }

  if (use_delta) {
    for (size_t i = 0; i < numpixels; i++) {
      img[i] = img[i] + delta_frame[i] - 32768u;
    }
  }

  return true;
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////

void ExtractFrame(const uint8_t* frame, size_t xsize, size_t ysize, int shift,
                  bool big_endian, uint16_t* out) {
  size_t numpixels = xsize * ysize;
  for (size_t i = 0; i < numpixels; i++) {
    uint8_t high = frame[i * 2 + (big_endian ? 0 : 1)];
    uint8_t low = frame[i * 2 + (big_endian ? 1 : 0)];
    out[i] = (high << 8u) | low;
    out[i] <<= shift;
  }
}

void UnextractFrame(const uint16_t* img, size_t xsize, size_t ysize, int shift,
                    bool big_endian, uint8_t* out) {
  size_t numpixels = xsize * ysize;
  for (size_t i = 0; i < numpixels; i++) {
    uint16_t u = img[i];
    u >>= shift;
    uint8_t a = u & 255;
    uint8_t b = u >> 8;
    if (big_endian) std::swap(a, b);
    out[i * 2 + 0] = a;
    out[i * 2 + 1] = b;
  }
}

////////////////////////////////////////////////////////////////////////////////

void StreamingDecoder::Decode(const uint8_t* bytes, size_t size,
    std::function<void(bool ok, uint16_t* frame, size_t xsize, size_t ysize,
        void* payload)> callback,
    void* payload) {
  // Don't copy the memory if not needed, but if the buffer already has data
  // we need to copy to get contiguous memory of all chunks.
  if (!buffer.empty()) {
    buffer.insert(buffer.end(), bytes, bytes + size);
  }

  const uint8_t* in = buffer.empty() ? bytes : buffer.data();
  size_t insize = buffer.empty() ? size : buffer.size();

  bool has_header = !delta_frame.empty();
  size_t pos = 0;
  if (delta_frame.empty() && insize > 12) {
    xsize = ReadUint32LE(in + 0);
    ysize = ReadUint32LE(in + 4);
    pos += 8;

    size_t deltasize = ReadUint32LE(in + pos);
    if (deltasize + pos <= insize) {
      delta_frame.resize(xsize * ysize);
      if (!DecompressFrame({}, in, insize, xsize, ysize, &pos,
          delta_frame.data())) {
        callback(FAILURE(), nullptr, 0, 0, payload);
      }
      has_header = true;
    } else {
      pos = 0;
    }
  }

  for (;;) {
    if (!has_header) break;
    if (pos + 5 > insize) break;

    size_t framesize = ReadUint32LE(in + pos);
    if (pos + framesize > insize) break;
    if (in[pos + 4] & 2) break;  // Frame index reached, end of frames.


    std::vector<uint16_t> frame(xsize * ysize);
    bool ok = DecompressFrame(delta_frame.data(), in, insize,
        xsize, ysize, &pos, frame.data());

    if (!ok) {
      callback(FAILURE(), nullptr, 0, 0, payload);
      return;
    }

    callback(ok, frame.data(), xsize, ysize, payload);
    id++;
  }

  if (buffer.empty()) {
    // Add any unprocessed bytes
    if (pos < size) {
      buffer.assign(bytes + pos, bytes + size);
    }
  } else {
    if (pos > 0) {
      if (pos == buffer.size()) {
        buffer.clear();
      } else {
        // Move the unprocessed data to the front
        memmove(buffer.data(), buffer.data() + pos, buffer.size() - pos);
        buffer.resize(buffer.size() - pos);
      }
    }
  }
}


////////////////////////////////////////////////////////////////////////////////

bool RandomAccessDecoder::Init(const uint8_t* data, size_t size) {
  if (size < 25) return FAILURE();  // Cannot contain header and footer.

  data_ = data;
  size_ = size;

  xsize_ = ReadUint32LE(data + 0);
  ysize_ = ReadUint32LE(data + 4);
  if (xsize_ == 0 || ysize_ == 0) return FAILURE();

  // Parse the delta frame
  size_t delta_frame_size = ReadUint32LE(data + 8);
  size_t pos = 8;
  delta_frame.resize(xsize_ * ysize_);
  if (!fpvc::DecompressFrame({}, data, size, xsize_, ysize_, &pos,
      delta_frame.data())) {
    return FAILURE();
  }
  if (delta_frame.size() != xsize_ * ysize_) return FAILURE();

  // Parse the frame index
  size_t num_frames = ReadUint64LE(data + size - 8);
  size_t footer_size = 5 + 8 * num_frames + 8;
  if (8 + delta_frame_size + footer_size > size) return FAILURE();
  frame_offsets.resize(num_frames);
  pos = size - footer_size;
  size_t verify_footer_size = ReadUint32LE(data + pos);
  if (verify_footer_size != footer_size) return FAILURE();
  if (data[pos + 4] != 2) return FAILURE();
  pos += 5;
  for (size_t i = 0; i < num_frames; i++) {
    frame_offsets[i] = ReadUint64LE(data + pos);
    pos += 8;
  }

  return true;
}

bool RandomAccessDecoder::DecodeFrame(size_t index, uint16_t* frame) const {
  if (index >= frame_offsets.size()) return FAILURE();
  size_t offset = frame_offsets[index];
  if (offset > size_) return FAILURE();
  size_t frame_size = ReadUint32LE(data_ + offset);
  if (offset + frame_size > size_) return FAILURE();
  return fpvc::DecompressFrame(delta_frame.data(),
      data_, size_, xsize_, ysize_, &offset, frame);
}



////////////////////////////////////////////////////////////////////////////////



Encoder::Encoder(size_t num_threads) {
  threads.resize(num_threads);
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i] = new std::thread(&Encoder::RunThread, this);
  }
}

void Encoder::Init(const uint16_t* delta_frame, size_t xsize, size_t ysize,
    Callback callback, void* payload) {
  delta_frame_.assign(delta_frame, delta_frame + xsize * ysize);
  xsize_ = xsize;
  ysize_ = ysize;
  std::vector<uint8_t> compressed;
  compressed.resize(8);
  WriteUint32LE(xsize, compressed.data() + 0);
  WriteUint32LE(ysize, compressed.data() + 4);
  fpvc::CompressFrame(nullptr, delta_frame, xsize, ysize, &compressed);
  bytes_written = compressed.size();
  callback(compressed.data(), compressed.size(), payload);
}

void Encoder::Finish(Callback callback, void* payload) {
  {
    std::unique_lock<std::mutex> l(m);
    if (finish) return;  // Already done.
    // Wait until everything is output.
    cv_main.wait(l, [this]{return q_out.empty();});
    finish = true;
  }
  cv_in.notify_all();

  for (size_t i = 0; i < threads.size(); i++) {
    threads[i]->join();
    delete threads[i];
  }

  std::vector<uint8_t> compressed;
  WriteFrameIndex(&compressed);
  callback(compressed.data(), compressed.size(), payload);
}

void Encoder::CompressFrame(const uint16_t* img,
    Callback callback, void* payload) {
  Task task;
  task.frame = img;
  task.id = id++;
  task.callback = callback;
  task.payload = payload;

  if (threads.empty()) {
    // Don't use multithreading
    std::vector<uint8_t> compressed = RunTask(task);
    FinishTask(task, &compressed);
    return;
  }

  {
    std::unique_lock<std::mutex> l(m);
    q_in.push(task);
    q_out.push(task);
  }
  cv_in.notify_all();

  {
    std::unique_lock<std::mutex> l(m);
    // Wait if the queue is too full so that only the maximum promised amount
    // of simultaneous tasks needing different input memory buffers is active
    // or queued.
    cv_main.wait(l, [this]{return q_out.size() < MaxQueued();});
  }
}

std::vector<uint8_t> Encoder::RunTask(const Task& task) {
  std::vector<uint8_t> compressed;
  fpvc::CompressFrame(delta_frame_.data(), task.frame,
      xsize_, ysize_, &compressed);
  return compressed;
}

size_t Encoder::MaxQueued() const {
  // The result must be at least as large as the amount of threads to be able
  // to use them all, must be at least 1 if there are no threads, and can be
  // made larger to potentially allow the main thread to queue in more input
  // data if the worker threads are all busy.
  return threads.empty() ? 1 : (threads.size() + (threads.size() + 1) / 2);
}

void Encoder::FinishTask(const Task& task, std::vector<uint8_t>* compressed) {
  frame_offsets.push_back(bytes_written);
  bytes_written += compressed->size();
  task.callback(compressed->data(), compressed->size(), task.payload);
}

void Encoder::WriteFrameIndex(std::vector<uint8_t>* compressed) const {
  size_t pos = compressed->size();
  size_t frameindex_size = 5 + 8 * frame_offsets.size() + 8;
  compressed->resize(compressed->size() + frameindex_size);
  WriteUint32LE(frameindex_size, &(*compressed)[pos]);
  (*compressed)[pos + 4] = 2;  // flags indicating it's the frame index
  pos += 5;
  for (size_t i = 0; i < frame_offsets.size(); i++) {
    WriteUint64LE(frame_offsets[i], &(*compressed)[pos]);
    pos += 8;
  }
  WriteUint64LE(frame_offsets.size(), &(*compressed)[pos]);
}

void Encoder::RunThread() {
  for (;;) {
    Task task;

    // Wait for starting a new compression
    {
      std::unique_lock<std::mutex> l(m);
      cv_in.wait(l, [this]{
        return !q_in.empty() || finish;
      });
      if (finish) return;
      task = q_in.front();
      q_in.pop();
    }

    std::vector<uint8_t> compressed = RunTask(task);

    // Wait to output in the correct order.
    {
      std::unique_lock<std::mutex> l(m);
      cv_out.wait(l, [&task, this]{
        return q_out.front().id == task.id;
      });
      FinishTask(task, &compressed);
      q_out.pop();
    }
    // Finished outputting
    cv_out.notify_all();
    // Changed q_out size
    cv_main.notify_one();
  }
}

}  // namespace fpvc

