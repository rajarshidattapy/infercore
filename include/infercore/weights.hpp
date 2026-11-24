// InferCore - weight file formats.
//
// ICW1 (fp32):  magic(4) num_tensors(u32)
//   per tensor: name_len(u32) name name_ndims(u32) dims(i64 x ndims)
//               dtype(u32, 0 = f32) data(f32 x numel)
// ICQ1 (int8):  same framing; dtype 1 = int8 data with per-channel f32 scales
//   (num_scales(u32) scales(f32 x n) precede the int8 payload).
#pragma once

#include <string>
#include <vector>

#include "infercore/tensor.hpp"

namespace infercore {

struct WeightTensor {
  std::string name;
  bool quantized = false;  // false: f32 holds data; true: q holds data
  Tensor f32;              // fp32 payload (or bias, in both formats)
  QWeights q;              // int8 payload with per-channel scales
};

std::vector<WeightTensor> load_weights_file(const std::string& path);
size_t file_size_bytes(const std::string& path);

}  // namespace infercore
