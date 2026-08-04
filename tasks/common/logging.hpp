#pragma once

#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <mutex>

namespace crane {

// Minimal thread-safe logger matching the Python logging format:
//   HH:MM:SS [LEVEL] message
class Logger {
 public:
  enum Level { kDebug = 0, kInfo = 1, kWarning = 2, kError = 3 };

  static Logger& instance() {
    static Logger logger;
    return logger;
  }

  void set_level(Level level) { level_ = level; }
  Level level() const { return level_; }

  void log(Level level, const char* fmt, ...) {
    if (level < level_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    char time_buf[16];
    std::time_t now = std::time(nullptr);
    std::tm tm_now;
    localtime_r(&now, &tm_now);
    std::strftime(time_buf, sizeof(time_buf), "%H:%M:%S", &tm_now);
    static const char* kNames[] = {"DEBUG", "INFO", "WARNING", "ERROR"};
    std::fprintf(stdout, "%s [%s] ", time_buf, kNames[level]);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stdout, fmt, args);
    va_end(args);
    std::fputc('\n', stdout);
    std::fflush(stdout);
  }

 private:
  Level level_ = kInfo;
  std::mutex mutex_;
};

}  // namespace crane

#define LOG_DEBUG(...) \
  ::crane::Logger::instance().log(::crane::Logger::kDebug, __VA_ARGS__)
#define LOG_INFO(...) \
  ::crane::Logger::instance().log(::crane::Logger::kInfo, __VA_ARGS__)
#define LOG_WARN(...) \
  ::crane::Logger::instance().log(::crane::Logger::kWarning, __VA_ARGS__)
#define LOG_ERROR(...) \
  ::crane::Logger::instance().log(::crane::Logger::kError, __VA_ARGS__)
