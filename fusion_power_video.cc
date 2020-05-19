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

A fusion power video file contains 1 or more grayscale 16-bit image frames. The
format contains one static delta frame used for compression of all frames,
and a frame index with pointers to all individual frames in the footer. Each
frame also contains a smaller preview image. To decode a frame, only the main
header, the one static delta frame and compressed data for the frame itself are
needed, frames do not depend on other frames.

Each format section below describes the concatenation of one or more streams or
values encoded in bytes (octets), possibly referring to sub-sections to further
define a stream, culminating in the full file format.

full file format:
-header: see header format below
-encoded delta frame: see delta frame format below
-one or more times: encoded frame: see frame format below
-footer: see footer format (frame index) below
Note: the encoder can only choose a single delta frame which can be used for
prediction of all the frames (so frames don't depend on each other, only on
the one delta frame); this image itself does not have to be the same as any
of the actual frames; the encoder should make a good choice of delta frame
for good compression (example: the first frame, assuming the camera remains
static and the first frame is a full representative image).

header format:
-4 bytes: xsize (little endian 32-bit integer)
-4 bytes: ysize (little endian 32-bit integer)

delta frame format:
-4 bytes: compressed size of entire frame, including these 4 bytes (little
 endian 32-bit integer)
-1 byte: chunk flags, must have value 1
-remaining bytes: the delta frame, encoded in the format described under image
 format below, without previous delta frame.

frame format:
-4 bytes: compressed size of entire frame, including these 4 bytes (little
 endian 32-bit integer), preview and main image
-1 byte: chunk flags, must have value 0
-4 bytes: preview_size: compressed size of preview image, excluding these 4
 bytes
-preview_size bytes: preview image in the image format, see image format below.
 This preview image is encoded without any delta frame, and its xsize and ysize
 are 1/8th of the respective main xsize and ysize, rounded up. The preview
 image should only use the 6 most significant bits of each 16-bit sample and
 leave all other 10 bits at zero.
-remaining bytes: the full frame encoded in the image format, see image format
 below

image format, given an xsize, ysize and an optional delta frame of the same
 dimensions:
-1 byte: image flags, see below
-variable amount of bytes: brotli compressed low bytes, or empty if not present
 (see flags)
-variable amount of bytes: brotli compressed high bytes
Note: the brotli decoder knows where the first brotli stream ends so the
split point is known during decoding. The brotli format is specified in
RFC 7932. See below for the complete procedure to decode an image.

footer format (frame index):
-4 bytes: size of this entire footer, including these 4 bytes (little
 endian 32-bit integer)
-1 byte: chunk flags, must have value 2
-per frame:
--8 bytes: offset from the start of the file to the start of this frame (little
  endian 64-bit integer)
-8 bytes: amount of frames
Note: if the full file is available, the decoder can compute the start of the
footer by parsing the last 8 bytes, rather than jumping frame by frame through
the entire file from the front, in case one wants to decode only a particular
frame.

chunk flags meanings:
-flags & 1: this must be true for the delta frame immediately after the header,
 and false for all other frames. Indicates this is not a frame to be decoded,
 but the delta frame that all other frames can use as base for prediction.
-flags & 2: this must be true for the footer (frame index), and must be false
 for all frames.

image flags meanings:
-flags & 1: if true, delta frame prediction is enabled. This must be false if
 this frame is the delta frame.
-flags & 2: if true, clamped gradient prediction is enabled
-flags & 4: if true, the compressed low bytes brotli stream is not present, all
 lower bytes are taken to be 0. Note: this can be used for the preview image.

procedure to decode an image:
-Note: given the xsize and ysize, a frame has xsize columns and ysize rows.
-brotli decompress the low bytes. These correspond to the LSB's of the 16-bit
 image. If this brotli stream is not present, set all xsize * ysize low bytes
 to 0 instead.
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
-if clamped gradient prediction is enabled, then for all pixels except those of
 the topmost row and the first column of the second row, compute: new_high_byte
 = old_high_byte + ClampedGradient(new_n, new_w, new_nw), with new_n, new_w and
 new_nw the already computed new values (or existing value if it was from top 
 row or first pixel of the second row) of the pixel respectively above, left,
 and above left of the current pixel. The addition should wrap on overflow. 
 ClampedGradient is defined mathematically (where the intermediate operations 
 happen in a space large enough to not overflow) as: clamp((n + w - nw), min(n,
 w, nw), max(n, w, nw)). The low bytes are not predicted as they 1. most often
 contain just noise and 2. because of their nature as lower half of a 16bit 
 value, the ClampedGradient is no valid predictor for them
-if delta prediction is enabled, then for all pixels compute:
 new_high_byte = old_high_byte + delta_frame_high_byte, and the same for the
 low byte plane (with delta_frame_value_high_byte the high byte value at the
 corresponding position from the delta frame). The addition and subtraction
 must wrap on overflow.
-combine the low and high byte streams into a single xsize*ysize 16-bit frame of
 unsigned 16-bit integers.
-the resulting 16-bit image may represent 12-bit data (or another bit amount),
 in that case, the least significant bits will be set to 0.
-Note: If this is a preview image, only 6 bits per sample are used and the other
 10 bits are zero. For displaying on an 8-bit display, repeat the two MSBs to
 fill in the two LSBs of the 8-bit display sample.
*/

namespace fpvc {
namespace {

// Enable this to print line number and message on decoding errors.
//#define FAIL_DEBUG_MESSAGE

// Prevent out of memory
#define MAX_IMAGE_SIZE 1000000000

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
  if (!enc) return FAILURE("couldn't init brotli encoder");
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
      return FAILURE("brotli compression failed");
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
  if (!decoder) return FAILURE("couldn't init brotli decoder");

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
  if (result != BROTLI_DECODER_RESULT_SUCCESS) {
    return FAILURE("brotli decompression failed");
  }
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
uint8_t ClampedGradient(uint8_t n, uint8_t w, uint8_t nw) {
  const uint8_t i = std::min(n, w), a = std::max(n, w);
  const uint8_t gradient = n + w - nw;
  const uint8_t clamped = (nw < i) ? a : gradient;
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

// Returns whether pos + width > size, taking overflow into account.
bool OutOfBounds(size_t pos, size_t width, size_t size) {
  return (pos > size) || (size - pos < width);
}

void CompressImage(const uint16_t* delta_frame,
                   const uint16_t* img,
                   size_t xsize, size_t ysize,
                   bool preview,
                   std::vector<uint8_t>* out) {
  int use_delta = 0;
  int use_clamped_gradient = 0;
  int zero_low = preview ? 1 : 0;
  size_t numpixels = xsize * ysize;
  std::vector<uint8_t> high;
  std::vector<uint8_t> low;

  if (delta_frame) {
    // heuristic to choose to use delta
    size_t skip = 15;  // let the heuristic run faster
    std::vector<size_t> counta(256);
    std::vector<size_t> countd(256);

    for (size_t i = 0; i < numpixels; i += skip) {
      uint8_t a = img[i] >> 8;
      uint8_t d = a - (delta_frame[i] >> 8);
      counta[a]++;
      countd[d]++;
    }

    if (EstimateEntropy(countd) < EstimateEntropy(counta)) use_delta = 1;

    if (use_delta) {
      high.reserve(numpixels);
      low.reserve(numpixels);
      for (size_t i = 0; i < numpixels; i++) {
        high.push_back(((img[i] >> 8) - (delta_frame[i] >> 8)) & 0xff);
        low.push_back(((img[i] &0xff) - (delta_frame[i] & 0xff)) & 0xff);
      }
    }
  }

  if (!use_delta) {
    high.reserve(numpixels);
    low.reserve(numpixels);
    for (size_t i = 0; i < numpixels; i++) {
      uint16_t pixel = img[i];
      high.push_back((pixel >> 8) & 0xff);
      low.push_back(pixel & 0xff);
    }
  }

  {
    std::vector<size_t> counta(256);
    std::vector<size_t> countb(256);

    size_t skip = 31;  // let the heuristic run faster
    for (size_t i = xsize + 1; i < numpixels; i += skip) {
      uint8_t a = high[i];
      uint8_t n = high[i - xsize];
      uint8_t w = high[i - 1];
      uint8_t nw = high[i - xsize - 1];
      uint8_t b = a - ClampedGradient(n, w, nw);
      counta[a]++;
      countb[b]++;
    }

    if (EstimateEntropy(countb) < EstimateEntropy(counta))
      use_clamped_gradient = 1;

    if (use_clamped_gradient) {
      std::vector<uint8_t> temp = high;
      for (size_t i = xsize + 1; i < numpixels; i++) {
          uint8_t n = high[i - xsize];
          uint8_t w = high[i - 1];
          uint8_t nw = high[i - xsize - 1];
          temp[i] = high[i] - ClampedGradient(n, w, nw);
      }
      high.swap(temp);
    }
  }

  std::vector<uint8_t> lowc;
  std::vector<uint8_t> highc;

  // NOTE: for this use case, brotli quality 1 gives smaller result than
  // brotli quality 2, yet is faster. Only the entropy coding matters, not the
  // LZ77.
  int quality = 1;
  if (!zero_low) {
    BrotliCompress(low.data(), low.size(), &lowc, quality);
  }
  BrotliCompress(high.data(), high.size(), &highc, quality);
  size_t result_size = 1 + highc.size() + lowc.size();
  out->resize(out->size() + result_size);
  uint8_t* out_data = out->data() + out->size() - result_size;
  uint8_t flags = use_delta + (use_clamped_gradient << 1) + (zero_low << 2);
  out_data[0] = flags;
  if (!zero_low) memcpy(out_data + 1, lowc.data(), lowc.size());
  memcpy(out_data + 1 + lowc.size(), highc.data(), highc.size());
}

bool DecompressImage(const uint16_t* delta_frame,
                     const uint8_t* in, size_t size,
                     size_t xsize, size_t ysize, uint16_t* img) {
  size_t pos = 0;
  if (pos >= size) return FAILURE("out of bounds");

  uint8_t flags = in[pos];
  bool use_delta = flags & 1;
  bool use_clamped_gradient = flags & 2;
  bool zero_low = flags & 4;
  (pos)++;
  if (!xsize || !ysize) return FAILURE("invalid image dimensions");
  size_t numpixels = xsize * ysize;
  // Error: want to use inter-frame delta but delta_frame frame not supplied.
  if (use_delta && !delta_frame) return FAILURE("delta frame not given");

  std::vector<uint8_t> low;
  if (zero_low) {
    low.resize(numpixels, 0);
  } else {
    if (!BrotliDecompress(in, size, &pos, &low)) return FAILURE();
  }

  std::vector<uint8_t> high;
  if (!BrotliDecompress(in, size, &pos, &high)) return FAILURE();

  // Error: sizes don't match image size
  if (low.size() != numpixels) return FAILURE("wrong decompressed plane size");
  if (high.size() != numpixels) return FAILURE("wrong decompressed plane size");

  if (use_clamped_gradient) {
    for (size_t i = xsize + 1; i < numpixels; i++) {
      uint8_t n = high[i - xsize];
      uint8_t w = high[i - 1];
      uint8_t nw = high[i - xsize - 1];
      high[i] = high[i] + ClampedGradient(n, w, nw);
    }
  }

  if (use_delta) {
    for (size_t i = 0; i < numpixels; i++) {
      img[i] = ((high[i] + (delta_frame[i] >> 8)) << 8) 
            | ((low[i] + (delta_frame[i] & 0xff)) & 0xff);
    }
  } else {
    for (size_t i = 0; i < numpixels; i++) {
      img[i] = (high[i] << 8) | low[i];
    }
  }

  return true;
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////


void CompressFrameWithPreview(const uint16_t* delta_frame,
                              const uint16_t* img,
                              size_t xsize, size_t ysize,
                              std::vector<uint8_t>* out) {
  // Downsample the preview frame to 1/8th width and height.
  static const int F = 8;  // downscale factor
  size_t xsize2 = (xsize + F - 1) / F;
  size_t ysize2 = (ysize + F - 1) / F;
  std::vector<uint16_t> preview(xsize2 * ysize2);
  for (size_t y = 0; y < ysize2; ++y) {
    for (size_t x = 0; x < xsize2; ++x) {
      uint32_t sum = 0;
      size_t x0 = x * F, y0 = y * F;
      size_t x1 = x0 + F, y1 = y0 + F;
      if (x1 > xsize) x1 = xsize;
      if (y1 > ysize) y1 = ysize;
      for (size_t y2 = y0; y2 < y1; ++y2) {
        for (size_t x2 = x0; x2 < x1; ++x2) {
          sum += img[y2 * xsize + x2];
        }
      }
      sum /= ((x1 - x0) * (y1 - y0));
      // Mask to take only the 6 MSBs
      preview[y * xsize2 + x] = sum & 0xfc00;
    }
  }

  size_t begin = out->size();
  out->resize(out->size() + 9);
  CompressImage(nullptr, preview.data(), xsize2, ysize2, false, out);
  size_t preview_size = out->size() - begin - 9;
  CompressImage(delta_frame, img, xsize, ysize, false, out);
  size_t total_size = out->size() - begin;
  WriteUint32LE(total_size, out->data() + begin);
  // Flag indicating this is not a delta frame or frame list.
  (*out)[begin + 4] = 0;
  WriteUint32LE(preview_size, out->data() + begin + 5);
}

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

  #define FAIL_CALLBACK(message) {\
    callback(FAILURE(message), nullptr, 0, 0, payload);\
    return;\
  }

  const uint8_t* in = buffer.empty() ? bytes : buffer.data();
  size_t insize = buffer.empty() ? size : buffer.size();

  bool has_header = !delta_frame.empty();
  size_t pos = 0;
  if (delta_frame.empty() && insize > 13) {
    xsize = ReadUint32LE(in + 0);
    ysize = ReadUint32LE(in + 4);
    pos += 8;

    if (xsize == 0 || ysize == 0) FAIL_CALLBACK("invalid image dimensions");
    if (xsize > 65536 || ysize > 65536 || xsize * ysize > MAX_IMAGE_SIZE) {
      // In theory larger sizes are possible, but this prevents OOM
      FAIL_CALLBACK("image too large");
    }

    size_t deltasize = ReadUint32LE(in + pos);
    if (deltasize < 5) FAIL_CALLBACK("too small for delta frame");
    uint8_t flag = in[pos + 4];
    if (flag != 1) FAIL_CALLBACK("not a delta frame");
    if (deltasize + pos <= insize) {
      delta_frame.resize(xsize * ysize);
      if (!DecompressImage({}, in + pos + 5, deltasize - 5, xsize, ysize,
          delta_frame.data())) {
        FAIL_CALLBACK("decompressing delta frame failed");
      }
      pos += deltasize;
      has_header = true;
    } else {
      pos = 0;
    }
  }

  for (;;) {
    if (!has_header) break;
    if (pos + 9 > insize) break;

    size_t frame_size = ReadUint32LE(in + pos);
    uint8_t flag = in[pos + 4];
    if (flag == 2) break;  // Frame index reached, end of frames.
    if (flag != 0) FAIL_CALLBACK("not a standard frame");
    if (pos + frame_size > insize) break;
    size_t preview_size = ReadUint32LE(in + pos + 5);
    if (preview_size > frame_size) FAIL_CALLBACK("preview size too large");

    size_t main_size = frame_size - preview_size - 9;
    std::vector<uint16_t> frame(xsize * ysize);
    bool ok = DecompressImage(delta_frame.data(), in + pos + 9 + preview_size, main_size,
        xsize, ysize, frame.data());
    pos += frame_size;

    if (!ok) FAIL_CALLBACK("decompressing frame failed");

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

  #undef FAIL_CALLBACK
}


////////////////////////////////////////////////////////////////////////////////

bool RandomAccessDecoder::Init(const uint8_t* data, size_t size) {
  // Header and delta frame size
  if (size < 12) return FAILURE("data too small to contain header");

  data_ = data;
  size_ = size;

  xsize_ = ReadUint32LE(data + 0);
  ysize_ = ReadUint32LE(data + 4);
  if (xsize_ == 0 || ysize_ == 0) return FAILURE("invalid image dimensions");
  if (xsize_ > 65536 || ysize_ > 65536 || xsize_ * ysize_ > MAX_IMAGE_SIZE) {
    // In theory larger sizes are possible, but this prevents OOM
    return FAILURE("image too large");
  }

  // Parse the delta frame
  size_t pos = 8;
  size_t delta_frame_size = ReadUint32LE(data + pos);
  if (OutOfBounds(pos, delta_frame_size, size)) return FAILURE("out of bounds");
  if (delta_frame_size < 5) return FAILURE("delta frame too small");
  uint8_t flag = data[12];
  if (flag != 1) return FAILURE("must begin with delta frame");
  delta_frame.resize(xsize_ * ysize_);
  if (!fpvc::DecompressImage({}, data + pos + 5, delta_frame_size - 5,
      xsize_, ysize_, delta_frame.data())) {
    return FAILURE("failed to decode delta frame");
  }
  pos += delta_frame_size;
  if (delta_frame.size() != xsize_ * ysize_) {
    return FAILURE("delta frame dimensions don't match");
  }

  // Parse the frame index
  size_t num_frames = ReadUint64LE(data + size - 8);
  // Prevent num_frames overflow, the entire file needs at least 16 bytes per
  // frame for its frame index listing and the frame's own header.
  if (num_frames > size / 16) return FAILURE("too many frames");
  size_t footer_size = 5 + 8 * num_frames + 8;
  if (footer_size > size) return FAILURE("footer too large");
  frame_offsets.resize(num_frames);
  pos = size - footer_size;
  size_t verify_footer_size = ReadUint32LE(data + pos);
  if (verify_footer_size != footer_size) {
    return FAILURE("footer size mismatch");
  }
  // Flag must be 2 to indicate frame index.
  if (data[pos + 4] != 2) return FAILURE("must end with frame index");
  pos += 5;
  for (size_t i = 0; i < num_frames; i++) {
    frame_offsets[i] = ReadUint64LE(data + pos);
    pos += 8;
  }

  return true;
}

bool RandomAccessDecoder::DecodeFrame(size_t index, uint16_t* frame) const {
  if (index >= frame_offsets.size()) return FAILURE("invalid frame index");
  size_t offset = frame_offsets[index];
  if (OutOfBounds(offset, 9, size_)) return FAILURE("out of bounds");
  const uint8_t* data = data_ + offset;

  size_t frame_size = ReadUint32LE(data);
  if (frame_size < 9) return FAILURE("frame too small");
  if (OutOfBounds(offset, frame_size, size_)) return FAILURE("out of bounds");
  uint8_t flag = data[4];
  if (flag != 0) return FAILURE("not a standard frame");
  size_t preview_size = ReadUint32LE(data + 5);
  if (preview_size > frame_size - 9) return FAILURE("preview too large");
  size_t main_size = frame_size - preview_size - 9;
  if (!fpvc::DecompressImage(delta_frame.data(),
      data + 9 + preview_size, main_size, xsize_, ysize_, frame)) {
    return FAILURE();
  }
  return true;
}

bool RandomAccessDecoder::DecodePreview(size_t index, uint8_t* preview) const {
  if (index >= frame_offsets.size()) return FAILURE("invalid preview index");

  size_t offset = frame_offsets[index];
  if (OutOfBounds(offset, 9, size_)) return FAILURE();
  const uint8_t* data = data_ + offset;

  size_t frame_size = ReadUint32LE(data);
  if (frame_size < 9) return FAILURE("frame too small");
  if (OutOfBounds(offset, frame_size, size_)) return FAILURE("out of bounds");
  uint8_t flag = data[4];
  if (flag != 0) return FAILURE("not a standard frame");
  size_t preview_size = ReadUint32LE(data + 5);
  if (OutOfBounds(9, preview_size, frame_size)) {
    return FAILURE("preview too large");
  }

  size_t xsize = preview_xsize();
  size_t ysize = preview_ysize();
  std::vector<uint16_t> preview16(xsize * ysize);
  if (!fpvc::DecompressImage(delta_frame.data(),
      data + 9, preview_size, xsize, ysize, preview16.data())) {
    return FAILURE("failed to decompress preview");
  }

  for (size_t y = 0; y < ysize; y++) {
    for (size_t x = 0; x < xsize; x++) {
      preview[y * xsize + x] = preview16[y * xsize + x] >> 8;
      // Since the preview is stored as 6-bit, repeat the MSBs in LSBs to
      // use all 8 bits.
      preview[y * xsize + x] |= (preview[y * xsize + x] >> 6);
    }
  }

  return true;
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
  compressed.resize(13);
  WriteUint32LE(xsize, compressed.data() + 0);
  WriteUint32LE(ysize, compressed.data() + 4);
  fpvc::CompressImage(nullptr, delta_frame, xsize, ysize, false, &compressed);
  WriteUint32LE(compressed.size() - 8, compressed.data() + 8);
  compressed[12] = 1;  // Flag indicating delta frame.
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
  fpvc::CompressFrameWithPreview(delta_frame_.data(), task.frame,
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
  (*compressed)[pos + 4] = 2;  // flag indicating it's the frame index
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

