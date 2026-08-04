#include "config.hpp"

#include <fstream>
#include <stdexcept>

#include "common/logging.hpp"

namespace crane {

void VisionConfig::load(const std::string& path) {
  std::lock_guard<std::mutex> lock(mutex_);
  root_ = YAML::LoadFile(path);  // throws YAML::BadFile on missing file
  path_ = path;
}

bool VisionConfig::save() {
  if (path_.empty()) return false;
  return save_as(path_);
}

bool VisionConfig::save_as(const std::string& path) {
  std::lock_guard<std::mutex> lock(mutex_);
  try {
    YAML::Emitter out;
    out << root_;
    std::ofstream fout(path, std::ios::binary | std::ios::trunc);
    if (!fout) {
      LOG_ERROR("Failed to open %s for writing", path.c_str());
      return false;
    }
    fout << out.c_str() << '\n';
    fout.flush();
    return static_cast<bool>(fout);
  } catch (const std::exception& e) {
    LOG_ERROR("Failed to save config %s: %s", path.c_str(), e.what());
    return false;
  }
}

std::map<std::string, int> VisionConfig::get_int_map(
    const std::vector<std::string>& path) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::map<std::string, int> result;
  YAML::Node node = root_;
  for (const auto& key : path) {
    if (!node || !node.IsMap()) return result;
    node.reset(node[key]);
  }
  if (!node || !node.IsMap()) return result;
  for (const auto& kv : node) {
    try {
      result[kv.first.as<std::string>()] = kv.second.as<int>();
    } catch (...) {
    }
  }
  return result;
}

}  // namespace crane
