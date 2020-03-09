## Fusion Power Video

Fusion power video is a fast lossless video compressor designed to compress
successive photographic 12-bit or 16-bit grayscale frames of the plasma inside a
fusion reactor in real-time, with simple predictors designed to work well on
different models of cameras with different noise and inter-frame behavior.

This is not an official Google product.

## Build Instructions

If needed, install libbrotli-dev and CMake:

`sudo apt-get install libbrotli-dev cmake`

Compile by creating a build directory and, from it, running:

```bash
cmake -DCMAKE_BUILD_TYPE=Release $source_directory
make
```
