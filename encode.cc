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
  size_t result = 0;
  std::istringstream sstream(s);
  sstream >> result;
  return result;
}


int main(int argc, char* argv[]) {
  if (argc < 5) {
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
  size_t num_threads = 4;
  if (argc > 5) {
    num_threads = ParseInt(argv[5]);
  }

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

  std::vector<uint8_t> buffer(framesize);


  fpvc::Encoder encoder(num_threads);

  bool initialized = false;

  // Callback function for the encoder: writes the data to stdout, and frees
  // allocated memory given by the payload, if any.
  auto WriteAndFree = [](const uint8_t* compressed, size_t size,
                         void* payload) {
    if (payload) {
      free(payload);
    }
    fwrite(compressed, 1, size, stdout);
  };

  while (std::cin) {
    if (!std::cin.read((char*)buffer.data(), framesize)) break;

    // The memory is allocated here, but freed in the callback, since each
    // thread needs its own memory. The callbacks are guaranteed to be called
    // after processing of that frame is completely finished.
    uint16_t* img = (uint16_t*)malloc(xsize * ysize * sizeof(uint16_t));
    fpvc::ExtractFrame(buffer.data(), xsize, ysize, shift, big_endian, img);

    if (!initialized) {
      initialized = true;
      encoder.Init(img, xsize, ysize, WriteAndFree, nullptr);
    }

    encoder.CompressFrame(img, WriteAndFree, img);
  }

  encoder.Finish(WriteAndFree, nullptr);
}
