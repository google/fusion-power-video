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

#include <fstream>
#include <iostream>
#include <sstream>

#include "fusion_power_video.h"

size_t ParseInt(const std::string& s) {
  size_t result;
  std::istringstream sstream(s);
  sstream >> result;
  return result;
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " xsize ysize shift big_endian < infile > outfile\n"
              << "    xsize, ysize: frame size in pixels\n"
              << "    big_endian: endianness of the raw input data, 0 or 1\n"
              << "    shift: how many bits to shift left to match MSBs, to"
              << " ensure the leftmost bits of uint16 are used for 12-bit data:"
              << " xxxxxxxxxxxx0000\n"
              << std::endl;
    return 1;
  }

  size_t xsize = ParseInt(argv[1]);
  size_t ysize = ParseInt(argv[2]);
  size_t big_endian = ParseInt(argv[3]);
  size_t shift = ParseInt(argv[4]);

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

  size_t framesize = xsize * ysize * 2;

  std::vector<uint16_t> prev;
  std::vector<uint16_t> img;
  std::vector<uint8_t> compressed;
  std::vector<uint8_t> buffer(framesize);

  while (std::cin) {
    if(!std::cin.read((char*)buffer.data(), framesize)) break;

    img = ExtractFrame(buffer.data(), xsize, ysize, shift, big_endian);
    compressed = CompressFrame(img, prev, xsize, ysize);
    fwrite(compressed.data(), 1, compressed.size(), stdout);

    prev.swap(img);
  }
}
