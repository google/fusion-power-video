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

// Benchmark and roundtrip test

#include <sys/time.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "fusion_power_video.h"

template <typename T>
std::string ToString(const T& val) {
  std::ostringstream sstream;
  sstream << val;
  return sstream.str();
}

size_t ParseInt(const std::string& s) {
  size_t result = 0;
  std::istringstream sstream(s);
  sstream >> result;
  return result;
}

std::vector<unsigned char> LoadFile(const std::string& filename,
    size_t maxsize) {
  std::ifstream f(filename, std::ios::binary);
  if (!f) return {};
  f.seekg(0, std::ios::end);
  size_t size = f.tellg();
  if (maxsize > 0 && size > maxsize) size = maxsize;
  f.seekg(0, std::ios::beg);
  std::vector<unsigned char> result(size);
  f.read(reinterpret_cast<char*>(result.data()), size);
  return result;
}

double GetTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

struct BenchmarkTime {
  BenchmarkTime() { start(); }

  void start() { time = GetTime(); }

  double stop() { return (GetTime() - time); }

  double time = 0;
};

void PrintBenchmark(const std::string& label, size_t pixels, size_t size,
                    double time = 0, size_t numframes = 0) {
  double speed = pixels / time / 1000000.0;
  double bpp = size / (double)pixels * 8;
  std::cerr << label << ": " << size << " bytes";
  if (pixels) {
    std::cerr << ", " << bpp << " bpp";
  }
  if (numframes > 1) {
    std::cerr << ", bytes per frame: " << ((double)size / numframes);
  }
  if (time > 0) {
    std::cerr << ", time: " << (time * 1000) << " ms"
              << ", speed: " << speed << " MP/s";
    std::cerr << ", frames per second: " << ((double)numframes / time);
  }
  std::cerr << std::endl;
}

// Renders a downscaled version of the preview in the terminal for testing.
void RenderPreview(const uint8_t* preview, size_t xsize, size_t ysize) {
  for(size_t y = 0; y < ysize; y += 4) {
    for(size_t x = 0; x < xsize; x += 4) {
      int v = preview[y * xsize + x];
      if(v < 16) std::cerr << ' ';
      else if(v < 24) std::cerr << '.';
      else if(v < 32) std::cerr << ',';
      else if(v < 48) std::cerr << ':';
      else if(v < 64) std::cerr << ';';
      else if(v < 128) std::cerr << '+';
      else if(v < 192) std::cerr << '=';
      else std::cerr << '#';
    }
    std::cerr << std::endl;
  }
  std::cerr << std::endl;
}

// Runs encoder benchmark and does roundtrip test with both the streaming
// decoder and the random access decoder.
void RunBenchmark(const std::string& filename,
                  size_t xsize, size_t ysize, int shift,
                  bool big_endian, size_t maxframes, size_t num_threads) {
  size_t maxsize = maxframes * xsize * ysize * 2;
  std::vector<unsigned char> raw = LoadFile(filename, maxsize);
  if (raw.empty()) {
    std::cerr << "couldn't load " << filename << std::endl;
    std::exit(1);
  }

  size_t framesize = xsize * ysize * 2;
  size_t numpixels = xsize * ysize;
  size_t num = raw.size() / framesize;

  if (num * framesize != raw.size()) {
    std::cerr << "raw filesize is not a multiple of framesize at 2 bytes per"
              << " pixel: " << framesize << std::endl;
  }

  maxframes = (maxframes == 0) ? num : (std::min(num, maxframes));

  size_t total_pixels = 0;
  BenchmarkTime total_timer;

  struct Frame {
    std::vector<uint8_t> orig;
    std::vector<uint8_t> compressed;
    size_t index;
  };
  std::vector<Frame> frames(maxframes);

  for (size_t i = 0; i < frames.size(); i++) {
    Frame& frame = frames[i];
    frame.orig.assign(raw.data() + framesize * i,
                      raw.data() + framesize * (i + 1));
    frame.index = i;
    total_pixels += numpixels;
  }

  uint16_t *delta_frame = reinterpret_cast<uint16_t*>(raw.data());

  std::vector<uint8_t> header;
  std::vector<uint8_t> footer;

  // Benchmark the encoder

  total_timer.start();
  {
    fpvc::Encoder encoder(num_threads, shift, big_endian);

    encoder.Init(delta_frame, xsize, ysize, [&header, numpixels](
        const uint8_t* compressed, size_t size, void* payload) {
      header.assign(compressed, compressed + size);
      PrintBenchmark("header", 0, size, 0);
    }, nullptr);
    for (size_t i = 0; i < frames.size(); i++) {
      Frame& frame = frames[i];
      uint16_t *frame_data = reinterpret_cast<uint16_t*>(raw.data() + framesize * i);
      encoder.CompressFrame(frame_data,
          [numpixels](const uint8_t* compressed, size_t size, void* payload) {
            Frame& frame = *reinterpret_cast<Frame*>(payload);
            frame.compressed.assign(compressed, compressed + size);
            PrintBenchmark(
                "frame " + ToString(frame.index), numpixels, size, 0);
          }, &frame);
    }
    encoder.Finish([&footer](
        const uint8_t* compressed, size_t size, void* payload) {
      footer.assign(compressed, compressed + size);
      PrintBenchmark("footer", 0, size, 0);
    }, nullptr);
  }
  double total_time = total_timer.stop();

  std::vector<uint8_t> compressed = header;
  for (size_t i = 0; i < frames.size(); i++) {
    compressed.insert(compressed.end(),
        frames[i].compressed.begin(), frames[i].compressed.end());
  }
  compressed.insert(compressed.end(), footer.begin(), footer.end());

  PrintBenchmark("total", total_pixels, compressed.size(), total_time,
      frames.size());

  // Test the streaming decoder
  {
    std::cerr << "verifying streaming decoder..." << std::endl;
    fpvc::StreamingDecoder decoder;
    size_t frames_decoded = 0;
    size_t blocksize = 65536;
    size_t pos = 0;
    while (pos < compressed.size()) {
      size_t size = (pos + blocksize > compressed.size()) ?
          (compressed.size() - pos) : blocksize;
      decoder.Decode(&compressed[pos], size,
          [shift, big_endian, framesize, &frames_decoded, &frames](
              bool ok, const uint16_t* image,
              size_t xsize, size_t ysize, void* payload) {
        if (!ok) {
          std::cerr << "StreamingDecoder decode failed" << std::endl;
          std::exit(1);
        }
        if (frames_decoded >= frames.size()) {
          std::cerr << "Too many frames" << std::endl;
          std::exit(1);
        }
        const Frame& frame = frames[frames_decoded++];
        const std::vector<uint8_t>& before = frame.orig;
        std::vector<uint8_t> after(framesize);
        fpvc::UnextractFrame(image, xsize, ysize, shift, big_endian,
                             after.data());
        if (before != after) {
          std::cerr << "Error: roundtrip not equal! " << frame.index << ": "
                    << after.size() << " " << frame.compressed.size()
                    << std::endl;
          std::exit(1);
        }
      });
      pos += size;
    }
    if (frames_decoded != frames.size()) {
      std::cerr << "Error: not all frames decoded: "
                << frames_decoded << " / " << frames.size() << std::endl;
      std::exit(1);
    } else {
      std::cerr << "ok" << std::endl;
    }
  }

  // Test the random access decoder
  {
    std::cerr << "verifying random access decoder..." << std::endl;
    fpvc::RandomAccessDecoder decoder;
    if (!decoder.Init(compressed.data(), compressed.size())) {
      std::cerr << "RandomAccessDecoder::Init failed" << std::endl;
      std::exit(1);
    }
    if (decoder.numframes() != frames.size() ||
        decoder.xsize() != xsize || decoder.ysize() != ysize) {
      std::cerr << "RandomAccessDecoder::Init mismatch" << std::endl;
      std::exit(1);
    }

    size_t pxsize = decoder.preview_xsize();
    size_t pysize = decoder.preview_ysize();

    for (size_t i = 0; i < frames.size(); i++) {
      Frame& frame = frames[i];
      const std::vector<uint8_t>& before = frame.orig;
      std::vector<uint16_t> image(xsize * ysize);
      if (!decoder.DecodeFrame(i, image.data())) {
        std::cerr << "RandomAccessDecoder::DecodeFrame failed" << std::endl;
        std::exit(1);
      }
      std::vector<uint8_t> preview(pxsize * pysize);
      if (!decoder.DecodePreview(i, preview.data())) {
        std::cerr << "RandomAccessDecoder::DecodePreview failed" << std::endl;
        std::exit(1);
      }

      // Uncomment this to manually verify previews. This prints a lot of lines
      // in the terminal.
      //RenderPreview(preview.data(), pxsize, pysize);

      std::vector<uint8_t> after(framesize);
      fpvc::UnextractFrame(image.data(), xsize, ysize, shift, big_endian,
                           after.data());
      if (before != after) {
        std::cerr << "Error: roundtrip not equal! " << frame.index << ": "
                  << after.size() << " " << frame.compressed.size()
                  << std::endl;
        std::exit(1);
      }
    }
    std::cerr << "ok" << std::endl;
  }
}

int main(int argc, char* argv[]) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0] << " "
              << "filename xsize ysize shift big_endian [maxframes] [threads]\n"
              << "    xsize, ysize: frame size in pixels\n"
              << "    big_endian: endianness of the raw input data, 0 or 1\n"
              << "    shift: how many bits to shift left to match MSBs, to"
              << " ensure the leftmost bits of uint16 are used for 12-bit data:"
              << " xxxxxxxxxxxx0000\n"
              << "    maxframes: optional, limit amount of frames to test\n"
              << std::endl;
    return 1;
  }

  std::string filename = argv[1];
  size_t xsize = ParseInt(argv[2]);
  size_t ysize = ParseInt(argv[3]);
  size_t big_endian = ParseInt(argv[4]);
  size_t shift = ParseInt(argv[5]);

  // There is no theoretical size limit, but this guards against invalid input
  // arguments.
  if (xsize == 0 || xsize > 65536 || ysize == 0 || ysize > 65536) {
    std::cerr << "invalid xsize, ysize: " << xsize << " " << ysize << std::endl;
    return 1;
  }
  if (shift > 16) {
    std::cerr << "invalid shift: " << shift << std::endl;
    return 1;
  }

  size_t maxframes = 0;
  size_t numthreads = 8;

  if (argc >= 7) maxframes = ParseInt(argv[6]);
  if (argc >= 8) numthreads = ParseInt(argv[7]);

  RunBenchmark(filename, xsize, ysize, shift, big_endian,
               maxframes, numthreads);
}
