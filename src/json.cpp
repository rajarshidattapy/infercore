#include "infercore/json.hpp"

#include <cctype>

namespace infercore {

namespace {

struct Parser {
  const char* p;
  const char* end;

  void skip_ws() {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
  }
  [[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("json parse error: " + msg);
  }
  char peek() {
    skip_ws();
    if (p >= end) fail("unexpected end of input");
    return *p;
  }
  char take() {
    const char c = peek();
    ++p;
    return c;
  }
  void expect(char c) {
    if (take() != c) fail(std::string("expected '") + c + "'");
  }

  JsonValue parse_value() {
    const char c = peek();
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') {
      JsonValue v;
      v.type = JsonValue::Type::kString;
      v.str = parse_string();
      return v;
    }
    if (c == 't' || c == 'f') return parse_bool();
    if (c == 'n') return parse_null();
    return parse_number();
  }

  JsonValue parse_object() {
    JsonValue v;
    v.type = JsonValue::Type::kObject;
    expect('{');
    if (peek() == '}') {
      ++p;
      return v;
    }
    for (;;) {
      std::string key = parse_string();
      expect(':');
      v.object.emplace(std::move(key), parse_value());
      const char c = take();
      if (c == '}') break;
      if (c != ',') fail("expected ',' or '}' in object");
    }
    return v;
  }

  JsonValue parse_array() {
    JsonValue v;
    v.type = JsonValue::Type::kArray;
    expect('[');
    if (peek() == ']') {
      ++p;
      return v;
    }
    for (;;) {
      v.array.push_back(parse_value());
      const char c = take();
      if (c == ']') break;
      if (c != ',') fail("expected ',' or ']' in array");
    }
    return v;
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    for (;;) {
      if (p >= end) fail("unterminated string");
      const char c = *p++;
      if (c == '"') break;
      if (c == '\\') {
        if (p >= end) fail("bad escape");
        const char e = *p++;
        switch (e) {
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case 'u': p += 4; out += '?'; break;  // not needed for model files
          default: out += e; break;
        }
      } else {
        out += c;
      }
    }
    return out;
  }

  JsonValue parse_number() {
    skip_ws();
    const char* start = p;
    if (p < end && (*p == '-' || *p == '+')) ++p;
    while (p < end && (std::isdigit(static_cast<unsigned char>(*p)) || *p == '.' ||
                       *p == 'e' || *p == 'E' || *p == '-' || *p == '+'))
      ++p;
    if (p == start) fail("expected a number");
    JsonValue v;
    v.type = JsonValue::Type::kNumber;
    v.number = std::stod(std::string(start, p));
    return v;
  }

  JsonValue parse_bool() {
    JsonValue v;
    v.type = JsonValue::Type::kBool;
    if (peek() == 't') {
      p += 4;
      v.boolean = true;
    } else {
      p += 5;
      v.boolean = false;
    }
    return v;
  }

  JsonValue parse_null() {
    p += 4;  // "null"
    JsonValue v;
    return v;
  }
};

}  // namespace

JsonValue parse_json(const std::string& text) {
  Parser parser{text.data(), text.data() + text.size()};
  JsonValue v = parser.parse_value();
  parser.skip_ws();
  return v;
}

}  // namespace infercore
