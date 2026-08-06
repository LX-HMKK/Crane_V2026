#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <string>

namespace crane {

// Minimal thread-safe logger matching the Python logging format:
//   HH:MM:SS [LEVEL] message
// Every line is mirrored to stdout (console) AND appended to a per-day file
// `<log_dir>/crane_YYYYMMDD.log` (directory auto-created; log_dir defaults to
// "logs" and is overridable via --log-dir). If the file cannot be opened the
// logger degrades to console-only and prints a one-line stderr notice.
class Logger {
 public:
  enum Level { kDebug = 0, kInfo = 1, kWarning = 2, kError = 3 };

  static Logger& instance() {
    static Logger logger;
    return logger;
  }

  void set_level(Level level) { level_ = level; }
  Level level() const { return level_; }

  void set_log_dir(const std::string& dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    log_dir_ = dir.empty() ? "logs" : dir;
    close_file_locked();  // next log() re-opens under the new dir
  }

  void log(Level level, const char* fmt, ...) {
    if (level < level_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    std::time_t now = std::time(nullptr);
    std::tm tm_now;
#ifdef _WIN32
    localtime_s(&tm_now, &now);
#else
    localtime_r(&now, &tm_now);
#endif
    char time_buf[16];
    std::strftime(time_buf, sizeof(time_buf), "%H:%M:%S", &tm_now);
    char date_buf[16];
    std::strftime(date_buf, sizeof(date_buf), "%Y%m%d", &tm_now);
    static const char* kNames[] = {"DEBUG", "INFO", "WARNING", "ERROR"};

    FILE* out = ensure_file_locked(date_buf);
    va_list file_args;
    va_start(file_args, fmt);
    if (out) {
      std::fprintf(out, "%s [%s] ", time_buf, kNames[level]);
      std::vfprintf(out, fmt, file_args);
      std::fputc('\n', out);
      std::fflush(out);
    }
    va_end(file_args);

    std::fprintf(stdout, "%s [%s] ", time_buf, kNames[level]);
    va_list out_args;
    va_start(out_args, fmt);
    std::vfprintf(stdout, fmt, out_args);
    va_end(out_args);
    std::fputc('\n', stdout);
    std::fflush(stdout);
  }

 private:
  // Re-open the daily file if the day changed or it is not open yet. Returns
  // nullptr when file logging is unavailable (log() then stays console-only).
  FILE* ensure_file_locked(const char* date_buf) {
    if (file_ && std::strcmp(date_buf, file_date_) == 0) return file_;
    close_file_locked();
    std::error_code ec;
    std::filesystem::create_directories(log_dir_, ec);
    std::filesystem::path p =
        std::filesystem::path(log_dir_) /
        ("crane_" + std::string(date_buf) + ".log");
    file_ = std::fopen(p.string().c_str(), "a");
    if (!file_) {
      std::fprintf(stderr, "[logger] cannot open %s\n", p.string().c_str());
      std::fflush(stderr);
      return nullptr;
    }
    std::snprintf(file_date_, sizeof(file_date_), "%s", date_buf);
    return file_;
  }

  void close_file_locked() {
    if (file_) {
      std::fclose(file_);
      file_ = nullptr;
    }
    file_date_[0] = '\0';
  }

  std::string log_dir_ = "logs";
  FILE* file_ = nullptr;
  char file_date_[16] = {0};
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
