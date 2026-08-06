#include "web/web_ui.hpp"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <thread>

#include <sys/socket.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "app.hpp"
#include "common/logging.hpp"
#include "web_assets.hpp"

namespace crane {

namespace {

// MJPEG frames are downscaled before JPEG encode: the browser scales them to
// fit the screen anyway (object-fit: contain), so sending full 1280px JPEGs
// only wastes encode CPU and decode jank on every slider drag.
constexpr int kMjpegMaxWidth = 960;

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c;
    }
  }
  return out;
}

bool send_all(int fd, const char* data, size_t n) {
  size_t sent = 0;
  while (sent < n) {
    ssize_t rc = ::send(fd, data + sent, n - sent, MSG_NOSIGNAL);
    if (rc < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    sent += static_cast<size_t>(rc);
  }
  return true;
}

// Editable field descriptor shared by GET /api/config and POST handling.
struct Field {
  std::string key;     // dotted path, e.g. "lines.bean.left_x"
  std::string label;
  std::string type;    // "int" | "float" | "bool"
  std::string view;    // "bean" | "digit" | "settings"
  double min = 0;
  double max = 0;
  double step = 1;
};

std::vector<std::string> split_key(const std::string& key) {
  std::vector<std::string> parts;
  size_t pos = 0;
  while (pos <= key.size()) {
    size_t dot = key.find('.', pos);
    if (dot == std::string::npos) {
      parts.push_back(key.substr(pos));
      break;
    }
    parts.push_back(key.substr(pos, dot - pos));
    pos = dot + 1;
  }
  return parts;
}

}  // namespace

WebUi::WebUi(CompetitionApp& app, const std::string& listen_addr, int port)
    : app_(app), server_(listen_addr, port) {
  register_routes();
}

WebUi::~WebUi() { stop(); }

void WebUi::start() { server_.start(); }
void WebUi::stop() {
  server_.stop();
  // The delayed-exit thread must finish before *this is destroyed.
  if (exit_thread_.joinable()) exit_thread_.join();
}

void WebUi::register_routes() {
  server_.route("GET", "/", [this](const auto& r) { return handle_index(r); });
  server_.route("GET", "/api/status",
                [this](const auto& r) { return handle_status(r); });
  server_.route("GET", "/api/config",
                [this](const auto& r) { return handle_get_config(r); });
  server_.route("POST", "/api/config",
                [this](const auto& r) { return handle_set_config(r); });
  server_.route("POST", "/api/debug",
                [this](const auto& r) { return handle_debug(r); });
  server_.route("POST", "/api/save",
                [this](const auto& r) { return handle_save(r); });
  server_.route("POST", "/api/exit",
                [this](const auto& r) { return handle_exit(r); });
  server_.route_stream("GET", "/stream.mjpeg",
                       [this](int fd, const auto& r) { handle_mjpeg(fd, r); });
}

// ── handlers ──────────────────────────────────────────────────────────

HttpServer::Response WebUi::handle_index(const HttpServer::Request&) {
  HttpServer::Response resp;
  resp.content_type = "text/html; charset=utf-8";
  resp.body = crane::web::kIndexHtml;
  return resp;
}

HttpServer::Response WebUi::handle_status(const HttpServer::Request&) {
  auto s = app_.status();
  std::string json = "{";
  json += "\"bean\":\"" + std::string(s.bean_online ? "ok" : "offline") + "\",";
  json += "\"digit\":\"" + std::string(s.digit_online ? "ok" : "offline") +
          "\",";
  json += "\"serial\":\"" +
          std::string(!s.serial_present
                          ? "na"
                          : (s.serial_connected ? "ok" : "offline")) +
          "\",";
  json += "\"start\":" + std::string(s.start_active ? "true" : "false") + ",";
  json += "\"debug\":" + std::string(s.debug_mode ? "true" : "false") + ",";
  json += "\"recognition\":" +
          std::string(s.recognition_active ? "true" : "false") + ",";
  char buf[128];
  std::snprintf(buf, sizeof(buf), "\"bean_fps\":%.2f,\"digit_fps\":%.2f,",
                s.bean_infer_fps, s.digit_infer_fps);
  json += buf;
  if (s.frame.has_value()) {
    json += "\"frame\":[";
    for (size_t i = 0; i < s.frame->size(); ++i) {
      json += std::to_string((*s.frame)[i]);
      if (i + 1 < s.frame->size()) json += ",";
    }
    json += "],";
  } else {
    json += "\"frame\":null,";
  }
  json += "\"reason\":\"" + json_escape(s.last_reason) + "\"";
  json += "}";
  HttpServer::Response resp;
  resp.content_type = "application/json";
  resp.body = json;
  return resp;
}

HttpServer::Response WebUi::handle_get_config(const HttpServer::Request&) {
  VisionConfig& cfg = app_.cfg();
  int max_x = std::max(cfg.get<int>({"cameras", "bean", "width"}, 640),
                       cfg.get<int>({"cameras", "digit", "width"}, 640));

  std::vector<Field> fields = {
      {"lines.bean.left_x", "Bean left x", "int", "bean", 0, (double)max_x, 1},
      {"lines.bean.right_x", "Bean right x", "int", "bean", 0, (double)max_x, 1},
      {"lines.digit.x1", "Digit x1", "int", "digit", 0, (double)max_x, 1},
      {"lines.digit.x2", "Digit x2", "int", "digit", 0, (double)max_x, 1},
      {"lines.digit.x3", "Digit x3", "int", "digit", 0, (double)max_x, 1},
      {"lines.digit.x4", "Digit x4", "int", "digit", 0, (double)max_x, 1},
      {"cameras.bean.auto_exposure", "Bean auto exposure", "bool", "settings", 0, 1, 1},
      {"cameras.bean.exposure", "Bean exposure", "float", "settings", 1, 10000, 1},
      {"cameras.digit.auto_exposure", "Digit auto exposure", "bool", "settings", 0, 1, 1},
      {"cameras.digit.exposure", "Digit exposure", "float", "settings", 1, 10000, 1},
      {"models.bean.conf_thres", "Bean conf", "float", "settings", 0.05, 0.95, 0.05},
      {"models.digit.conf_thres", "Digit conf", "float", "settings", 0.05, 0.95, 0.05},
      {"logic.max_inferred_beans", "Bean infer max", "int", "settings", 0, 3, 1},
      {"logic.max_inferred_digits", "Digit infer max", "int", "settings", 0, 5, 1},
  };

  std::string json = "{\"max_x\":" + std::to_string(max_x) + ",\"fields\":[";
  bool first = true;
  for (const auto& f : fields) {
    auto path = split_key(f.key);
    std::string value;
    if (f.type == "bool") {
      value = cfg.get<bool>(path, false) ? "true" : "false";
    } else if (f.type == "int") {
      value = std::to_string(cfg.get<int>(path, 0));
    } else {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.4g", cfg.get<double>(path, 0.0));
      value = buf;
    }
    if (!first) json += ",";
    first = false;
    json += "{\"key\":\"" + f.key + "\",\"label\":\"" + f.label +
            "\",\"type\":\"" + f.type + "\",\"view\":\"" + f.view +
            "\",\"min\":" + std::to_string(f.min) +
            ",\"max\":" + std::to_string(f.max) +
            ",\"step\":" + std::to_string(f.step) +
            ",\"value\":" + value + "}";
  }
  json += "]}";
  HttpServer::Response resp;
  resp.content_type = "application/json";
  resp.body = json;
  return resp;
}

HttpServer::Response WebUi::handle_set_config(const HttpServer::Request& req) {
  auto params = req.query;
  if (params.count("key") == 0 && !req.body.empty()) {
    params = parse_query(req.body);
  }
  auto kit = params.find("key");
  auto vit = params.find("value");
  if (kit == params.end() || vit == params.end()) {
    return {400, "text/plain; charset=utf-8", "missing key/value"};
  }
  const std::string& key = kit->second;
  const std::string& raw = vit->second;
  auto path = split_key(key);

  VisionConfig& cfg = app_.cfg();
  try {
    // Coerce by existing value type (bool when the key holds one already,
    // int when integral, else float).
    if (raw == "true" || raw == "false") {
      cfg.set(path, raw == "true");
    } else {
      double v = std::stod(raw);
      if (raw.find('.') == std::string::npos &&
          raw.find('e') == std::string::npos) {
        cfg.set(path, static_cast<int>(v));
      } else {
        cfg.set(path, v);
      }
    }
  } catch (const std::exception& e) {
    return {400, "text/plain; charset=utf-8",
            std::string("bad value: ") + e.what()};
  }
  return {200, "application/json", "{\"ok\":true}"};
}

HttpServer::Response WebUi::handle_debug(const HttpServer::Request& req) {
  // Require an explicit mode: a bare POST (or a JSON body, which parse_query
  // cannot decode) must not silently force recognition on — that would bypass
  // the START safety gate. The bundled UI always sends ?mode=0|1.
  auto mit = req.query.find("mode");
  if (mit == req.query.end()) {
    return {400, "text/plain; charset=utf-8", "missing mode=0|1 parameter"};
  }
  bool on = !(mit->second == "0" || mit->second == "off" ||
              mit->second == "false");
  app_.set_debug_mode(on);
  HttpServer::Response resp;
  resp.content_type = "application/json";
  resp.body = std::string(on ? "{\"ok\":true,\"debug\":true}"
                             : "{\"ok\":true,\"debug\":false}");
  return resp;
}

HttpServer::Response WebUi::handle_save(const HttpServer::Request&) {
  bool ok = app_.save_cfg();
  if (!ok) {
    return {500, "application/json", "{\"ok\":false}"};
  }
  LOG_INFO("Configuration saved to %s", app_.cfg().path().c_str());
  return {200, "application/json", "{\"ok\":true}"};
}

HttpServer::Response WebUi::handle_exit(const HttpServer::Request&) {
  // Joinable (never detached): WebUi::stop() joins it, so the thread can
  // never dereference a destroyed WebUi/app. request_exit() only sets
  // flags, so joining after it returns is immediate.
  if (exit_thread_.joinable()) exit_thread_.join();
  exit_thread_ = std::thread([this] {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    app_.request_exit();
  });
  return {200, "application/json", "{\"ok\":true}"};
}

void WebUi::handle_mjpeg(int fd, const HttpServer::Request& req) {
  Channel ch = Channel::kBean;
  auto it = req.query.find("cam");
  if (it != req.query.end() && it->second == "digit") ch = Channel::kDigit;

  std::string head =
      "HTTP/1.0 200 OK\r\n"
      "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
      "Cache-Control: no-store\r\nConnection: close\r\n\r\n";
  if (!send_all(fd, head.data(), head.size())) return;

  std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};
  while (server_.server_running()) {
    cv::Mat view;
    if (!app_.render_view(ch, view) || view.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }
    std::vector<uint8_t> jpg;
    cv::Mat frame_out = view;
    if (view.cols > kMjpegMaxWidth) {
      double scale = static_cast<double>(kMjpegMaxWidth) / view.cols;
      cv::resize(view, frame_out,
                 cv::Size(kMjpegMaxWidth, static_cast<int>(view.rows * scale)),
                 0, 0, cv::INTER_AREA);
    }
    try {
      cv::imencode(".jpg", frame_out, jpg, params);
    } catch (const cv::Exception&) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }
    std::string chunk_head = "--frame\r\nContent-Type: image/jpeg\r\n"
                             "Content-Length: " +
                             std::to_string(jpg.size()) + "\r\n\r\n";
    if (!send_all(fd, chunk_head.data(), chunk_head.size())) return;
    if (!send_all(fd, reinterpret_cast<const char*>(jpg.data()), jpg.size()))
      return;
    if (!send_all(fd, "\r\n", 2)) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(80));  // ~12 fps
  }
}

}  // namespace crane
