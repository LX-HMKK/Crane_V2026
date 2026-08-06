// crane_vision — Crane 视觉运行时 (C++ port of scripts/main_amd64.py +
// scripts/main_arm64.py). Dual camera → dual YOLO11 inference → logic
// completion → 12-byte serial frames; browser-based calibration UI.
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "app.hpp"
#include "io/camera.hpp"
#include "common/logging.hpp"
#include "common/sockets.hpp"

namespace {

std::atomic<bool> g_stop{false};

void on_signal(int) { g_stop = true; }

void print_usage(const char* argv0) {
  std::printf(
      "Usage: %s [options]\n"
      "\n"
      "Modes (default: competition loop with web UI)\n"
      "  --once                single scan on the inputs, print result, exit\n"
      "  --bench N             benchmark N inference rounds per model, exit\n"
      "  --list-cameras        list V4L2 cameras, exit\n"
      "\n"
      "Options\n"
      "  --config PATH         vision.yaml path (default configs/vision.yaml)\n"
      "  --backend NAME        auto | onnx | bpu (default auto)\n"
      "  --mock-cameras        synthetic test cameras\n"
      "  --bean-image PATH     static bean camera image (overrides camera)\n"
      "  --digit-image PATH    static digit camera image (overrides camera)\n"
      "  --no-serial           disable the serial thread\n"
      "  --no-web              disable the web calibration UI\n"
      "  --web-port N          web UI port (default 8080)\n"
      "  --web-listen ADDR     web UI listen address (default 0.0.0.0)\n"
      "  --ort-threads N       ONNX Runtime intra-op threads (default auto)\n"
      "  -v, --verbose         debug logging\n"
      "  -h, --help            this help\n",
      argv0);
}

}  // namespace

int main(int argc, char** argv) {
  crane::net::Wsa wsa;  // no-op on POSIX; required before any socket call
  crane::AppOptions opts;
  std::string config_path = "configs/vision.yaml";
  std::string backend = "auto";
  bool once = false;
  bool list_cams = false;
  int bench_rounds = 0;
  bool verbose = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--config") {
      config_path = next("--config");
    } else if (arg == "--backend") {
      backend = next("--backend");
    } else if (arg == "--mock-cameras") {
      opts.mock_cameras = true;
    } else if (arg == "--bean-image") {
      opts.bean_image = next("--bean-image");
    } else if (arg == "--digit-image") {
      opts.digit_image = next("--digit-image");
    } else if (arg == "--no-serial") {
      opts.use_serial = false;
    } else if (arg == "--no-web") {
      opts.web_enabled = false;
    } else if (arg == "--web-port") {
      opts.web_port = std::stoi(next("--web-port"));
    } else if (arg == "--web-listen") {
      opts.web_listen = next("--web-listen");
    } else if (arg == "--ort-threads") {
      opts.ort_threads = std::stoi(next("--ort-threads"));
    } else if (arg == "--once") {
      once = true;
    } else if (arg == "--bench") {
      bench_rounds = std::stoi(next("--bench"));
    } else if (arg == "--list-cameras") {
      list_cams = true;
    } else if (arg == "-v" || arg == "--verbose") {
      verbose = true;
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      print_usage(argv[0]);
      return 2;
    }
  }

  crane::Logger::instance().set_level(verbose ? crane::Logger::kDebug
                                              : crane::Logger::kInfo);

  if (list_cams) {
    crane::list_cameras();
    return 0;
  }

  opts.backend = backend;
  // Static-image / once / bench modes never touch hardware serial or web.
  if (once || bench_rounds > 0) {
    opts.use_serial = false;
    opts.web_enabled = false;
  }

  crane::VisionConfig cfg;
  try {
    cfg.load(config_path);
  } catch (const std::exception& e) {
    LOG_ERROR("Failed to load config %s: %s", config_path.c_str(), e.what());
    return 1;
  }

  try {
    crane::CompetitionApp app(std::move(cfg), opts);
    if (once) return app.run_once();
    if (bench_rounds > 0) return app.bench(bench_rounds);

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
#ifndef _WIN32
    std::signal(SIGPIPE, SIG_IGN);  // SIGPIPE is undefined on MSVC
#endif

    // Watchdog-style runner: run() blocks until stop() (signal or web EXIT).
    std::thread watcher([&] {
      while (!g_stop) std::this_thread::sleep_for(std::chrono::milliseconds(50));
      app.stop();
    });
    int rc = 1;
    try {
      rc = app.run();
    } catch (...) {
      // Join before unwinding — destroying a joinable thread terminates.
      g_stop = true;
      watcher.join();
      throw;
    }
    g_stop = true;
    watcher.join();
    return rc;
  } catch (const std::exception& e) {
    LOG_ERROR("Fatal: %s", e.what());
    return 1;
  }
}
