# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.6)

find_package(PkgConfig REQUIRED)
#pkg_check_modules(Brotli REQUIRED IMPORTED_TARGET libbrotlienc libbrotlidec)

add_library(fpv_columnar_batch STATIC columnar_batch.h columnar_batch.cc columnar_batch_encoder.h columnar_batch_encoder.cc columnar_batch_decoder.h columnar_batch_decoder.cc)
target_link_libraries(fpv_columnar_batch PRIVATE fusion_power_video pthread PkgConfig::Brotli)

set_target_properties(fpv_columnar_batch PROPERTIES PUBLIC_HEADER "columnar_batch.h;columnar_batch_encoder.h;columnar_batch_decoder.h")
INSTALL(TARGETS fpv_columnar_batch
        ARCHIVE DESTINATION lib 
        PUBLIC_HEADER DESTINATION include
)

foreach (executable IN ITEMS columnar_batch_encoder_test columnar_batch_decoder_test)
  add_executable("${executable}" "${executable}.cc")
  target_link_libraries("${executable}" fpv_columnar_batch)
endforeach ()
