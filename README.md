## Fusion Power Video

Fusion power video is a fast lossless video compressor designed to compress
successive photographic 12-bit or 16-bit grayscale frames of the plasma inside a
fusion reactor in real-time, with simple predictors designed to work well on
different models of cameras with different noise and inter-frame behavior.

## Build Instructions

If needed, install libbrotli-dev:

`sudo apt-get install libbrotli-dev`

Compile with:

`clang++ -O3 -std=c++11 benchmark.cc fusion_power_video.cc -lbrotlidec -lbrotlienc -o benchmark`

`clang++ -O3 -std=c++11 encode.cc fusion_power_video.cc -lbrotlidec -lbrotlienc -o encode`

`clang++ -O3 -std=c++11 decode.cc fusion_power_video.cc -lbrotlidec -lbrotlienc -o decode`

