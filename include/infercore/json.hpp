// InferCore - minimal JSON parser (objects, arrays, strings, numbers, bools, null).
// Just enough to load model graph descriptions without external dependencies.
#pragma once

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace infercore {

class JsonValue {
 public:
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

  Type type = Type::kNull;
  bool boolean = false;
  double number = 0.0;
  std::string str;
  std::vector<JsonValue> array;
  std::map<std::string, JsonValue> object;

  bool is_null() const { return type == Type::kNull; }
  const JsonValue& at(const std::string& key) const {
    auto it = object.find(key);
    if (it == object.end()) throw std::out_of_range("json: missing key " + key);
    return it->second;
  }
  const JsonValue* find(const std::string& key) const {
    auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
  }
  double get_number(const std::string& key, double fallback) const {
    const JsonValue* v = find(key);
    return v ? v->number : fallback;
  }
  std::string get_string(const std::string& key, const std::string& fallback) const {
    const JsonValue* v = find(key);
    return v ? v->str : fallback;
  }
  int64_t get_int(const std::string& key, int64_t fallback) const {
    return static_cast<int64_t>(get_number(key, static_cast<double>(fallback)));
  }
};

JsonValue parse_json(const std::string& text);

}  // namespace infercore
