#pragma once

#include <map>
#include <string>

namespace crane {

// 豆子类别名 → 协议颜色编码 (yellow→1, green→2, white→3 by default).
// Mirrors models/mapping.py.
inline const std::map<std::string, int> kDefaultBeanToDigit = {
    {"yellow", 1},
    {"green", 2},
    {"white", 3},
};

// Returns the mapped code, or -1 when unmapped (Python returns None).
int bean_name_to_digit(const std::string& name,
                       const std::map<std::string, int>& mapping);

}  // namespace crane
