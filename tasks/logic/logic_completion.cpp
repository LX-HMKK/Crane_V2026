#include "logic_completion.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace crane {

std::map<int, ZoneResult> validate_and_complete(
    const std::map<int, std::optional<std::pair<int, float>>>& zone_vals,
    const std::set<int>& required_set, float /*confidence_threshold*/) {
  // Zones and required values in sorted order (std::map/set iterate sorted,
  // matching Python's sorted()).
  std::vector<int> zones;
  zones.reserve(zone_vals.size());
  for (const auto& kv : zone_vals) zones.push_back(kv.first);
  std::vector<int> required(required_set.begin(), required_set.end());

  if (zones.size() != required.size()) {
    throw std::invalid_argument("zone count " + std::to_string(zones.size()) +
                                " must equal required count " +
                                std::to_string(required.size()));
  }

  // score[zone][value] = confidence or 0.0
  std::map<int, std::map<int, float>> score;
  std::set<int> empty_zones;
  for (int z : zones) {
    for (int v : required) score[z][v] = 0.0f;
    const auto& entry = zone_vals.at(z);
    if (!entry.has_value()) {
      empty_zones.insert(z);
      continue;
    }
    auto [val, conf] = *entry;
    if (required_set.count(val)) {
      score[z][val] = conf;
    } else {
      empty_zones.insert(z);
    }
  }

  // Enumerate all permutations of required values across zones in
  // lexicographic order — identical to itertools.permutations(sorted).
  // First best wins ties (strict `>` comparison), as in Python.
  std::vector<int> perm = required;
  std::vector<int> best_perm;
  float best_total = -1.0f;
  std::set<int> best_detected;

  do {
    float total = 0.0f;
    for (size_t i = 0; i < zones.size(); ++i) total += score[zones[i]][perm[i]];
    if (total > best_total) {
      best_total = total;
      best_perm = perm;
      best_detected.clear();
      for (size_t i = 0; i < zones.size(); ++i) {
        int z = zones[i];
        if (score[z][perm[i]] > 0.0f && !empty_zones.count(z)) {
          best_detected.insert(z);
        }
      }
    }
  } while (std::next_permutation(perm.begin(), perm.end()));

  std::map<int, ZoneResult> results;
  for (size_t i = 0; i < zones.size(); ++i) {
    int z = zones[i];
    ZoneResult& r = results[z];
    r.value = best_perm[i];
    if (best_detected.count(z)) {
      r.inferred = false;
      const auto& entry = zone_vals.at(z);
      r.conf = entry.has_value() ? std::optional<float>(entry->second)
                                 : std::nullopt;
    } else {
      r.inferred = true;
      r.conf = std::nullopt;
    }
  }
  return results;
}

int count_inferred(const std::map<int, ZoneResult>& results) {
  int n = 0;
  for (const auto& kv : results) n += kv.second.inferred ? 1 : 0;
  return n;
}

}  // namespace crane
