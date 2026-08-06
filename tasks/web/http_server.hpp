#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "common/sockets.hpp"

namespace crane {

// Minimal embedded HTTP/1.0 server (cross-platform sockets, thread per
// connection). Just enough for the calibration web UI: static responses,
// JSON APIs and MJPEG streaming. No external dependencies.
//
// Lifecycle: stop() shuts down the listen socket AND every tracked client
// connection, then waits for connection threads to drain, so streaming
// (MJPEG) handlers never touch application state after stop() returns.
class HttpServer {
 public:
  struct Request {
    std::string method;                            // GET / POST
    std::string path;                              // /api/config
    std::map<std::string, std::string> query;      // parsed query string
    std::map<std::string, std::string> headers;    // lower-cased keys
    std::string body;                              // raw request body
  };

  struct Response {
    int status = 200;
    std::string content_type = "text/plain; charset=utf-8";
    std::string body;
  };

  using Handler = std::function<Response(const Request&)>;
  // Streaming handler: writes the full response itself (headers included).
  // Must periodically check `server_running()` (or send failure) and return.
  using StreamHandler = std::function<void(net::Socket fd, const Request&)>;

  HttpServer(std::string listen_addr, int port);
  ~HttpServer();

  void route(const std::string& method, const std::string& path,
             Handler handler);
  void route_stream(const std::string& method, const std::string& path,
                    StreamHandler handler);

  void start();
  void stop();
  int port() const { return port_; }
  bool server_running() const { return running_; }

 private:
  void accept_loop();
  void handle_connection(net::Socket fd);  // exception-safe wrapper
  // Read + dispatch one request from the connection buffer. Returns false when
  // the connection must close (client Connection: close, stream route done,
  // malformed request, read error, or server stopping).
  bool handle_one_request(net::Socket fd, std::string& buf);

  std::string listen_addr_;
  int port_;
  net::Socket listen_fd_ = net::kInvalidSocket;
  std::atomic<bool> running_{false};
  std::thread accept_thread_;

  // Live client connections, drained on stop().
  std::mutex conn_mutex_;
  std::set<net::Socket> conn_fds_;
  std::atomic<int> active_conns_{0};

  std::map<std::string, Handler> handlers_;
  std::map<std::string, StreamHandler> stream_handlers_;
};

// URL-decode (percent + '+') helper exposed for handlers.
std::string url_decode(const std::string& s);
std::map<std::string, std::string> parse_query(const std::string& s);

}  // namespace crane
