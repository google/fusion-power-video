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
  size_t result;
  std::istringstream sstream(s);
  sstream >> result;
  return result;
}

// NOTE: For testing only, does not check if file is bounded.
std::vector<unsigned char> LoadFile(const std::string& filename) {
  std::ifstream f(filename, std::ios::binary);
  if (!f) return {};
  f.seekg(0, std::ios::end);
  size_t size = f.tellg();
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
                    double time) {
  double speed = pixels / time / 1000000.0;
  double bpp = size / (double)pixels * 8;
  std::cout << label << ": " << size << " bytes, " << bpp
            << " bpp, time: " << (time * 1000) << " ms"
            << ", speed: " << speed << " MP/s" << std::endl;
}

void TestSet(const std::string& filename, size_t xsize, size_t ysize, int shift,
             bool big_endian, size_t maxframes) {
  BenchmarkTime t;

  std::vector<unsigned char> raw = LoadFile(filename);
  if (raw.empty()) {
    std::cout << "couldn't load " << filename << std::endl;
    std::exit(1);
  }

  size_t framesize = xsize * ysize * 2;
  size_t num = raw.size() / framesize;

  if (num * framesize != raw.size()) {
    std::cerr << "raw filesize is not a multiple of framesize at 2 bytes per"
              << " pixel: " << framesize << std::endl;
  }

  std::vector<uint16_t> prev;

  maxframes = (maxframes == 0) ? num : (std::min(num, maxframes));

  size_t total_size = 0;
  size_t total_pixels = 0;
  double total_time = 0;

  std::vector<uint16_t> img;
  std::vector<uint8_t> compressed;

  for (size_t i = 0; i < maxframes; i++) {
    t.start();
    img = ExtractFrame(raw.data() + framesize * i, xsize, ysize, shift,
                       big_endian);
    compressed = CompressFrame(img, prev, xsize, ysize);
    double time = t.stop();
    PrintBenchmark("frame " + ToString(i), xsize * ysize, compressed.size(),
                   time);

    total_pixels += xsize * ysize;
    total_size += compressed.size();
    total_time += time;

    // verify
    {
      std::vector<uint8_t> before = std::vector<uint8_t>(
          raw.data() + framesize * i, raw.data() + framesize * i + framesize);
      size_t pos = 0;
      size_t expected_size =
          ReadVarint(compressed.data(), compressed.size(), &pos);
      std::vector<uint16_t> decompressed =
          DecompressFrame(prev, compressed.data(), compressed.size(), &pos);
      std::vector<uint8_t> after =
          UnextractFrame(decompressed, xsize, ysize, shift, big_endian);
      if (before != after) {
        std::cout << "Error: roundtrip not equal! " << decompressed.size()
                  << " " << after.size() << std::endl;
        std::exit(1);
      }
    }

    // comment out to NOT do delta compression between previous frame
    prev = img;
  }

  PrintBenchmark("total", total_pixels, total_size, total_time);
}

int main(int argc, char* argv[]) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0] << " "
              << "filename xsize ysize shift big_endian [maxframes]\n"
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

  if (argc >= 7) maxframes = ParseInt(argv[6]);

  TestSet(filename, xsize, ysize, shift, big_endian, maxframes);
}
