#include "mapping.hpp"

namespace crane {

int bean_name_to_digit(const std::string& name,
                       const std::map<std::string, int>& mapping) {
  auto it = mapping.find(name);
  return it == mapping.end() ? -1 : it->second;
}

}  // namespace crane
