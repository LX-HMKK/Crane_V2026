#pragma once

#include <memory>
#include <string>

#include "web/http_server.hpp"

namespace crane {

class CompetitionApp;

// Browser-based calibration UI: embedded fullscreen page + MJPEG previews
// + REST config/status/save endpoints. Replaces the Tkinter
// CalibrationUI/CompetitionUI from the Python runtime.
class WebUi {
 public:
  WebUi(CompetitionApp& app, const std::string& listen_addr, int port);
  ~WebUi();

  void start();
  void stop();

 private:
  void register_routes();

  HttpServer::Response handle_index(const HttpServer::Request&);
  HttpServer::Response handle_status(const HttpServer::Request&);
  HttpServer::Response handle_get_config(const HttpServer::Request&);
  HttpServer::Response handle_set_config(const HttpServer::Request&);
  HttpServer::Response handle_save(const HttpServer::Request&);
  HttpServer::Response handle_exit(const HttpServer::Request&);
  void handle_mjpeg(int fd, const HttpServer::Request&);

  CompetitionApp& app_;
  HttpServer server_;
  std::thread exit_thread_;  // joinable; joined in stop()
};

}  // namespace crane
