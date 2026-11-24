#include <chrono>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "infercore/json.hpp"
#include "infercore/model.hpp"
#include "infercore/op.hpp"
#include "infercore/thread_pool.hpp"
#include "infercore/weights.hpp"

namespace infercore {

namespace {

struct Node {
  std::string name;
  std::vector<std::string> inputs;  // upstream node names, or the graph input name
  std::unique_ptr<Op> op;
  std::string weights_name;
  std::string bias_name;
};

std::unique_ptr<Op> make_op(const JsonValue& layer, const std::string& name) {
  const std::string type = layer.at("type").str;
  if (type == "Conv2D")
    return make_conv2d(name, static_cast<int>(layer.get_int("stride", 1)),
                       static_cast<int>(layer.get_int("pad", 0)));
  if (type == "ReLU") return make_relu(name);
  if (type == "MaxPool")
    return make_maxpool(name, static_cast<int>(layer.get_int("kernel", 2)),
                        static_cast<int>(layer.get_int("stride", 2)));
  if (type == "Flatten") return make_flatten(name);
  if (type == "Gemm") return make_gemm(name);
  if (type == "Softmax") return make_softmax(name);
  throw std::runtime_error("unknown layer type: " + type);
}

class NativeModel final : public IModel {
 public:
  NativeModel(const std::string& graph_path, const std::string& weights_path,
              Precision precision, unsigned num_threads)
      : precision_(precision), pool_(num_threads) {
    std::ifstream is(graph_path);
    if (!is) throw std::runtime_error("cannot open graph json: " + graph_path);
    std::stringstream ss;
    ss << is.rdbuf();
    const JsonValue root = parse_json(ss.str());

    input_name_ = root.get_string("input_name", "input");
    for (const auto& d : root.at("input_shape").array)
      input_shape_.push_back(static_cast<int64_t>(d.number));

    // Build nodes in file order.
    std::vector<Node> ordered;
    std::string prev;
    for (const auto& layer : root.at("layers").array) {
      Node node;
      node.name = layer.at("name").str;
      node.op = make_op(layer, node.name);
      if (const JsonValue* ins = layer.find("inputs")) {
        for (const auto& i : ins->array) node.inputs.push_back(i.str);
      } else {
        node.inputs.push_back(prev.empty() ? input_name_ : prev);
      }
      if (node.op->needs_weights()) {
        node.weights_name = layer.at("weights").str;
        node.bias_name = layer.at("bias").str;
      }
      prev = node.name;
      ordered.push_back(std::move(node));
    }
    if (ordered.empty()) throw std::runtime_error("model has no layers");
    output_layer_ = root.get_string("output_layer", prev);

    // Topological levels (Kahn's algorithm) so independent nodes can be
    // scheduled onto the thread pool together.
    std::unordered_map<std::string, size_t> index;
    for (size_t i = 0; i < ordered.size(); ++i) index[ordered[i].name] = i;
    std::vector<size_t> depth(ordered.size(), 0);
    bool progress = true;
    size_t resolved = 0;
    // Iterative longest-path depth; fine for small graphs.
    while (progress) {
      progress = false;
      for (size_t i = 0; i < ordered.size(); ++i) {
        size_t d = 0;
        bool ready = true;
        for (const auto& up : ordered[i].inputs) {
          if (up == input_name_) continue;
          auto it = index.find(up);
          if (it == index.end())
            throw std::runtime_error("node " + ordered[i].name +
                                     " depends on unknown input " + up);
          if (depth[it->second] == 0 && it->second != i && ordered[it->second].inputs.size() > 0) {
            // not necessarily unresolved; handled by fixpoint below
          }
          d = std::max(d, depth[it->second] + 1);
        }
        if (d != depth[i]) {
          depth[i] = d;
          progress = true;
        }
        (void)ready;
      }
    }
    // Cycle check: if any depth exceeds node count, there is a cycle.
    size_t max_depth = 0;
    for (size_t d : depth) max_depth = std::max(max_depth, d);
    if (max_depth >= ordered.size())
      throw std::runtime_error("model graph has a cycle");
    levels_.resize(max_depth + 1);
    for (size_t i = 0; i < ordered.size(); ++i) levels_[depth[i]].push_back(i);
    nodes_ = std::move(ordered);
    (void)resolved;

    // Attach weights.
    std::unordered_map<std::string, WeightTensor> weights;
    for (auto& wt : load_weights_file(weights_path)) weights.emplace(wt.name, std::move(wt));
    for (auto& node : nodes_) {
      if (!node.op->needs_weights()) continue;
      auto wit = weights.find(node.weights_name);
      auto bit = weights.find(node.bias_name);
      if (wit == weights.end())
        throw std::runtime_error("weights file is missing tensor " + node.weights_name);
      if (bit == weights.end())
        throw std::runtime_error("weights file is missing tensor " + node.bias_name);
      if (precision_ == Precision::kFp32) {
        if (wit->second.quantized)
          throw std::runtime_error("fp32 model given an int8 weights file");
        node.op->set_weights(wit->second.f32, bit->second.f32);
      } else {
        if (!wit->second.quantized)
          throw std::runtime_error("int8 model given an fp32 weights file");
        node.op->set_weights_q(wit->second.q, bit->second.f32);
      }
    }

    // Warm-up run on a zero input: determines the output shape up front and
    // validates that the whole graph executes with the attached weights.
    Tensor zeros(input_shape_);
    run(zeros);
  }

  Tensor run(const Tensor& input) override {
    if (input.shape() != input_shape_)
      throw std::runtime_error("input shape mismatch");
    timings_.assign(nodes_.size(), OpTiming{});
    if (precision_ == Precision::kFp32)
      return run_fp32(input);
    return run_int8(input);
  }

  std::vector<int64_t> output_shape() const override { return output_shape_; }
  std::vector<int64_t> input_shape() const override { return input_shape_; }
  std::string backend() const override { return "native"; }
  std::string precision() const override {
    return precision_ == Precision::kFp32 ? "fp32" : "int8";
  }
  const std::vector<OpTiming>& last_timings() const override { return timings_; }

 private:
  Tensor run_fp32(const Tensor& input) {
    std::unordered_map<std::string, Tensor> arena;
    arena.reserve(nodes_.size() + 1);
    for (const auto& n : nodes_) arena.emplace(n.name, Tensor{});
    arena[input_name_] = input;
    for (const auto& level : levels_) {
      auto run_node = [&](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i) {
          Node& node = nodes_[level[i]];
          std::vector<const Tensor*> ins;
          for (const auto& up : node.inputs) ins.push_back(&arena.at(up));
          const auto t0 = std::chrono::steady_clock::now();
          Tensor out = node.op->forward(ins, pool_);
          const auto t1 = std::chrono::steady_clock::now();
          timings_[level[i]] = {node.name, node.op->type(),
                                std::chrono::duration<double, std::micro>(t1 - t0).count()};
          arena[node.name] = std::move(out);
        }
      };
      if (level.size() > 1)
        pool_.parallel_for(level.size(), 1, run_node);
      else
        run_node(0, level.size());
    }
    output_shape_ = arena.at(output_layer_).shape();
    return arena.at(output_layer_);
  }

  Tensor run_int8(const Tensor& input) {
    std::unordered_map<std::string, QTensor> arena;
    arena.reserve(nodes_.size() + 1);
    for (const auto& n : nodes_) arena.emplace(n.name, QTensor{});
    arena[input_name_] = quantize_dynamic(input);
    for (const auto& level : levels_) {
      auto run_node = [&](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i) {
          Node& node = nodes_[level[i]];
          std::vector<const QTensor*> ins;
          for (const auto& up : node.inputs) ins.push_back(&arena.at(up));
          const auto t0 = std::chrono::steady_clock::now();
          QTensor out = node.op->forward_q(ins, pool_);
          const auto t1 = std::chrono::steady_clock::now();
          timings_[level[i]] = {node.name, node.op->type(),
                                std::chrono::duration<double, std::micro>(t1 - t0).count()};
          arena[node.name] = std::move(out);
        }
      };
      if (level.size() > 1)
        pool_.parallel_for(level.size(), 1, run_node);
      else
        run_node(0, level.size());
    }
    Tensor out = dequantize(arena.at(output_layer_));
    output_shape_ = out.shape();
    return out;
  }

  Precision precision_;
  ThreadPool pool_;
  std::vector<Node> nodes_;
  std::vector<std::vector<size_t>> levels_;
  std::string input_name_;
  std::string output_layer_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<OpTiming> timings_;
};

}  // namespace

std::unique_ptr<IModel> load_native_model(const std::string& graph_json,
                                          const std::string& weights_path,
                                          Precision precision, unsigned num_threads) {
  return std::make_unique<NativeModel>(graph_json, weights_path, precision, num_threads);
}

}  // namespace infercore
