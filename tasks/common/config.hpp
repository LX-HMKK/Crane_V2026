#pragma once

#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace crane {

// Typed read/write accessor for configs/vision.yaml — the single source of
// truth, mirroring models/runtime.py load_vision_cfg + ConfigBinding.
//
// Path-based access: get<T>({"lines","bean","left_x"}, default).
// set() mutates the in-memory document and fires registered change listeners;
// save() persists the whole document back to disk (insertion order kept,
// matching yaml.dump(sort_keys=False)).
class VisionConfig {
 public:
  VisionConfig() = default;
  explicit VisionConfig(const std::string& path) { load(path); }

  VisionConfig(const VisionConfig&) = delete;
  VisionConfig& operator=(const VisionConfig&) = delete;
  VisionConfig(VisionConfig&& other) noexcept {
    std::lock_guard<std::mutex> lock(other.mutex_);
    path_ = std::move(other.path_);
    root_ = std::move(other.root_);
    listeners_ = std::move(other.listeners_);
  }
  VisionConfig& operator=(VisionConfig&& other) noexcept {
    if (this != &other) {
      std::scoped_lock lock(mutex_, other.mutex_);
      path_ = std::move(other.path_);
      root_ = std::move(other.root_);
      listeners_ = std::move(other.listeners_);
    }
    return *this;
  }

  void load(const std::string& path);
  bool save();  // write back to path_; returns false on IO error
  bool save_as(const std::string& path);

  const std::string& path() const { return path_; }
  YAML::Node root() const { return root_; }  // copy of handle (node is shared)

  // ── typed getters (path segments, default on any missing key) ──────
  // NOTE: yaml-cpp Node::operator= assigns CONTENT into the referenced
  // node (it is not handle rebind); use reset() when walking.
  template <typename T>
  T get(const std::vector<std::string>& path, const T& fallback) const {
    std::lock_guard<std::mutex> lock(mutex_);
    YAML::Node node = root_;
    for (const auto& key : path) {
      if (!node || !node.IsMap()) return fallback;
      node.reset(node[key]);
    }
    if (!node) return fallback;
    try {
      return node.as<T>();
    } catch (...) {
      return fallback;
    }
  }

  // String-keyed map getter: e.g. mapping.bean_to_digit.
  std::map<std::string, int> get_int_map(
      const std::vector<std::string>& path) const;

  // ── mutation ────────────────────────────────────────────────────────
  // set() writes the value and fires listeners registered for `path`
  // (exact path match). Returns the coerced value actually stored.
  template <typename T>
  void set(const std::vector<std::string>& path, const T& value) {
    std::vector<std::function<void()>> fired;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      YAML::Node node = root_;
      for (size_t i = 0; i + 1 < path.size(); ++i) {
        node.reset(node[path[i]]);  // yaml-cpp auto-vivifies map entries
      }
      node[path.back()] = value;
      fired.reserve(listeners_.count(path));
      auto range = listeners_.equal_range(path);
      for (auto it = range.first; it != range.second; ++it) {
        fired.push_back(
            [fn = it->second, node = YAML::Node(value)]() { fn(node); });
      }
    }
    for (auto& fn : fired) fn();
  }

  // Register an on-change callback (ConfigBinding.on_change equivalent).
  template <typename T>
  void on_change(const std::vector<std::string>& path,
                 std::function<void(const T&)> callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    listeners_.emplace(path, [cb = std::move(callback)](const YAML::Node& v) {
      try {
        cb(v.as<T>());
      } catch (...) {
      }
    });
  }

  std::mutex& mutex() { return mutex_; }

 private:
  std::string path_;
  YAML::Node root_;
  mutable std::mutex mutex_;
  std::multimap<std::vector<std::string>,
                std::function<void(const YAML::Node&)>>
      listeners_;
};

}  // namespace crane
