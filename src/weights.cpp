#include "infercore/weights.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace infercore {

namespace {

template <typename T>
T read_raw(std::istream& is) {
  T v;
  is.read(reinterpret_cast<char*>(&v), sizeof(T));
  if (!is) throw std::runtime_error("weights: unexpected end of file");
  return v;
}

}  // namespace

std::vector<WeightTensor> load_weights_file(const std::string& path) {
  std::ifstream is(path, std::ios::binary);
  if (!is) throw std::runtime_error("weights: cannot open " + path);
  char magic[4];
  is.read(magic, 4);
  if (!is) throw std::runtime_error("weights: bad file " + path);
  const bool is_fp32 = std::memcmp(magic, "ICW1", 4) == 0;
  const bool is_int8 = std::memcmp(magic, "ICQ1", 4) == 0;
  if (!is_fp32 && !is_int8)
    throw std::runtime_error("weights: bad magic in " + path);

  const uint32_t num_tensors = read_raw<uint32_t>(is);
  std::vector<WeightTensor> out;
  out.reserve(num_tensors);
  for (uint32_t t = 0; t < num_tensors; ++t) {
    WeightTensor wt;
    const uint32_t name_len = read_raw<uint32_t>(is);
    wt.name.resize(name_len);
    is.read(wt.name.data(), name_len);
    const uint32_t ndims = read_raw<uint32_t>(is);
    std::vector<int64_t> dims(ndims);
    for (uint32_t i = 0; i < ndims; ++i) dims[i] = read_raw<int64_t>(is);
    const uint32_t dtype = read_raw<uint32_t>(is);
    size_t numel = 1;
    for (int64_t d : dims) numel *= static_cast<size_t>(d);
    if (dtype == 0) {
      std::vector<float> data(numel);
      is.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(numel * 4));
      if (!is) throw std::runtime_error("weights: truncated tensor " + wt.name);
      wt.f32 = Tensor(std::move(dims), std::move(data));
      wt.quantized = false;
    } else if (dtype == 1) {
      const uint32_t num_scales = read_raw<uint32_t>(is);
      wt.q.scales.resize(num_scales);
      is.read(reinterpret_cast<char*>(wt.q.scales.data()),
              static_cast<std::streamsize>(num_scales * 4));
      wt.q.shape = std::move(dims);
      wt.q.data.resize(numel);
      is.read(reinterpret_cast<char*>(wt.q.data.data()),
              static_cast<std::streamsize>(numel));
      if (!is) throw std::runtime_error("weights: truncated tensor " + wt.name);
      wt.quantized = true;
    } else {
      throw std::runtime_error("weights: unknown dtype for " + wt.name);
    }
    out.push_back(std::move(wt));
  }
  return out;
}

size_t file_size_bytes(const std::string& path) {
  std::ifstream is(path, std::ios::binary | std::ios::ate);
  if (!is) throw std::runtime_error("cannot stat " + path);
  return static_cast<size_t>(is.tellg());
}

}  // namespace infercore
