#include "http_server.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "common/logging.hpp"

namespace crane {

std::string url_decode(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '%' && i + 2 < s.size()) {
      auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
      };
      int hi = hex(s[i + 1]), lo = hex(s[i + 2]);
      if (hi >= 0 && lo >= 0) {
        out += static_cast<char>(hi * 16 + lo);
        i += 2;
        continue;
      }
    }
    out += s[i] == '+' ? ' ' : s[i];
  }
  return out;
}

std::map<std::string, std::string> parse_query(const std::string& s) {
  std::map<std::string, std::string> out;
  size_t pos = 0;
  while (pos <= s.size()) {
    size_t amp = s.find('&', pos);
    std::string pair = s.substr(
        pos, amp == std::string::npos ? std::string::npos : amp - pos);
    if (!pair.empty()) {
      size_t eq = pair.find('=');
      if (eq != std::string::npos) {
        out[url_decode(pair.substr(0, eq))] = url_decode(pair.substr(eq + 1));
      } else {
        out[url_decode(pair)] = "";
      }
    }
    if (amp == std::string::npos) break;
    pos = amp + 1;
  }
  return out;
}

HttpServer::HttpServer(std::string listen_addr, int port)
    : listen_addr_(std::move(listen_addr)), port_(port) {}

HttpServer::~HttpServer() { stop(); }

void HttpServer::route(const std::string& method, const std::string& path,
                       Handler handler) {
  handlers_[method + " " + path] = std::move(handler);
}

void HttpServer::route_stream(const std::string& method,
                              const std::string& path,
                              StreamHandler handler) {
  stream_handlers_[method + " " + path] = std::move(handler);
}

void HttpServer::start() {
  listen_fd_ = net::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ == net::kInvalidSocket) {
    throw std::runtime_error("http server: socket() failed: " +
                             net::error_message(errno));
  }
  int one = 1;
  net::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port_));
  if (listen_addr_.empty() || listen_addr_ == "0.0.0.0") {
    addr.sin_addr.s_addr = INADDR_ANY;
  } else if (net::inet_pton(AF_INET, listen_addr_.c_str(), &addr.sin_addr) !=
             1) {
    net::close(listen_fd_);
    listen_fd_ = net::kInvalidSocket;
    throw std::runtime_error("http server: bad listen addr " + listen_addr_);
  }
  if (net::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) <
      0) {
    int e = errno;
    net::close(listen_fd_);
    listen_fd_ = net::kInvalidSocket;
    throw std::runtime_error("http server: bind() failed: " +
                             net::error_message(e));
  }
  if (net::listen(listen_fd_, 16) < 0) {
    net::close(listen_fd_);
    listen_fd_ = net::kInvalidSocket;
    throw std::runtime_error("http server: listen() failed: " +
                             net::error_message(errno));
  }
  running_ = true;
  accept_thread_ = std::thread(&HttpServer::accept_loop, this);
  LOG_INFO("Web UI listening on http://%s:%d",
           listen_addr_.empty() ? "0.0.0.0" : listen_addr_.c_str(), port_);
}

void HttpServer::stop() {
  running_ = false;
  if (listen_fd_ != net::kInvalidSocket) {
    net::shutdown(listen_fd_, net::kShutRdwr);
    net::close(listen_fd_);
    listen_fd_ = net::kInvalidSocket;
  }
  if (accept_thread_.joinable()) accept_thread_.join();
  // Shut down all live connections so streaming handlers (MJPEG) fail their
  // next send and exit, then wait for connection threads to drain.
  {
    std::lock_guard<std::mutex> lock(conn_mutex_);
    for (net::Socket fd : conn_fds_) net::shutdown(fd, net::kShutRdwr);
  }
  for (int i = 0; i < 100 && active_conns_ > 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  {
    std::lock_guard<std::mutex> lock(conn_mutex_);
    for (net::Socket fd : conn_fds_) net::close(fd);
    conn_fds_.clear();
  }
}

void HttpServer::accept_loop() {
  while (running_) {
    sockaddr_in client{};
    int len = sizeof(client);
    net::Socket fd =
        net::accept(listen_fd_, reinterpret_cast<sockaddr*>(&client), &len);
    if (fd == net::kInvalidSocket) {
      if (!running_) return;
      if (errno == EINTR) continue;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    // Flush small responses immediately (Nagle + delayed ACK would otherwise
    // stall them on the board's LAN too). NOTE: the ~200ms slider lag seen in
    // WSL2 is the localhost NAT's per-CONNECTION cost, fixed by keep-alive in
    // handle_connection() — TCP_NODELAY alone does not fix that case.
    int one = 1;
    net::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    {
      std::lock_guard<std::mutex> lock(conn_mutex_);
      conn_fds_.insert(fd);
    }
    ++active_conns_;
    std::thread(&HttpServer::handle_connection, this, fd).detach();
  }
}

namespace {

bool send_all(net::Socket fd, const char* data, size_t n) {
  size_t sent = 0;
  while (sent < n) {
    net::ssize_t rc = net::send(fd, data + sent, n - sent, net::kMsgNoSignal);
    if (rc < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    sent += static_cast<size_t>(rc);
  }
  return true;
}

}  // namespace

void HttpServer::handle_connection(net::Socket fd) {
  // RAII: always deregister on exit so stop() can drain cleanly. The guard
  // closes the fd only if it was still registered (stop() may have closed
  // it while draining).
  struct ConnGuard {
    HttpServer* srv;
    net::Socket fd;
    ~ConnGuard() {
      bool should_close;
      {
        std::lock_guard<std::mutex> lock(srv->conn_mutex_);
        should_close = srv->conn_fds_.erase(fd) > 0;
      }
      if (should_close) net::close(fd);
      --srv->active_conns_;
    }
  } guard{this, fd};

  // Any exception here would escape into a detached thread → std::terminate.
  // Malformed requests (bad content-length, ...) must just close quietly.
  try {
    // Keep-alive: serve multiple requests per connection. The WSL2 localhost
    // path charges ~200ms per new TCP connection; reusing one connection turns
    // slider/config POSTs from ~200ms into ~1ms. stop() shuts every fd down,
    // so a thread blocked on recv() here returns as soon as running_ flips.
    std::string buf;
    bool keep_alive = true;
    while (keep_alive && running_) {
      keep_alive = handle_one_request(fd, buf);
    }
  } catch (const std::exception& e) {
    LOG_WARN("http: request handling failed: %s", e.what());
  } catch (...) {
    LOG_WARN("http: unknown error in request handling");
  }
}

bool HttpServer::handle_one_request(net::Socket fd, std::string& buf) {
  // Read until a full header block is buffered (partial reads persist across
  // keep-alive iterations).
  size_t header_end = std::string::npos;
  while ((header_end = buf.find("\r\n\r\n")) == std::string::npos) {
    if (buf.size() >= 65536) return false;  // oversize
    char tmp[4096];
    net::ssize_t n = net::recv(fd, tmp, sizeof(tmp), 0);
    if (n <= 0) return false;  // EOF / error / fd shut down
    buf.append(tmp, static_cast<size_t>(n));
  }

  Request req;
  // Request line
  size_t line_end = buf.find("\r\n");
  if (line_end == std::string::npos || line_end > header_end) return false;
  std::string request_line = buf.substr(0, line_end);
  {
    size_t sp1 = request_line.find(' ');
    size_t sp2 = request_line.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
    req.method = request_line.substr(0, sp1);
    std::string target = request_line.substr(sp1 + 1, sp2 - sp1 - 1);
    size_t q = target.find('?');
    req.path = url_decode(q == std::string::npos ? target
                                                 : target.substr(0, q));
    if (q != std::string::npos) {
      req.query = parse_query(target.substr(q + 1));
    }
  }
  // Headers
  size_t pos = line_end + 2;
  while (pos < header_end) {
    size_t eol = buf.find("\r\n", pos);
    std::string line = buf.substr(pos, eol - pos);
    size_t colon = line.find(':');
    if (colon != std::string::npos) {
      std::string key = line.substr(0, colon);
      std::string value = line.substr(colon + 1);
      std::transform(key.begin(), key.end(), key.begin(), ::tolower);
      if (!value.empty() && value.front() == ' ') value.erase(0, 1);
      req.headers[key] = value;
    }
    pos = eol + 2;
  }

  // Body (content-length), consumed from the buffer.
  size_t consumed = header_end + 4;
  auto it = req.headers.find("content-length");
  if (it != req.headers.end()) {
    size_t want = static_cast<size_t>(std::stoul(it->second));
    while (buf.size() < consumed + want) {
      if (buf.size() >= 65536) return false;  // oversize
      char tmp[4096];
      net::ssize_t n = net::recv(fd, tmp, sizeof(tmp), 0);
      if (n <= 0) return false;
      buf.append(tmp, static_cast<size_t>(n));
    }
    req.body = buf.substr(consumed, want);
    consumed += want;
  }
  // Drop the consumed request; keep any pipelined bytes for the next pass.
  buf.erase(0, consumed);

  const std::string key = req.method + " " + req.path;

  auto sit = stream_handlers_.find(key);
  if (sit != stream_handlers_.end()) {
    sit->second(fd, req);  // handler streams until stop()/client disconnect
    return false;          // the stream owns this connection to completion
  }

  Response resp;
  auto hit = handlers_.find(key);
  if (hit != handlers_.end()) {
    try {
      resp = hit->second(req);
    } catch (const std::exception& e) {
      resp.status = 500;
      resp.content_type = "text/plain; charset=utf-8";
      resp.body = std::string("internal error: ") + e.what();
    }
  } else {
    resp.status = 404;
    resp.body = "not found";
  }

  // Keep-alive unless the client asked to close. Browsers reuse the pooled
  // connection, so only the first request per host pays connection setup.
  bool client_close = false;
  auto cit = req.headers.find("connection");
  if (cit != req.headers.end()) {
    std::string v = cit->second;
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    client_close = v.find("close") != std::string::npos;
  }
  bool keep_alive = !client_close && running_;

  static const std::map<int, std::string> kStatus = {
      {200, "200 OK"},
      {400, "400 Bad Request"},
      {404, "404 Not Found"},
      {405, "405 Method Not Allowed"},
      {500, "500 Internal Server Error"},
  };
  auto sit2 = kStatus.find(resp.status);
  std::string head = std::string(keep_alive ? "HTTP/1.1 " : "HTTP/1.0 ") +
                     (sit2 != kStatus.end() ? sit2->second : "500") +
                     "\r\nContent-Type: " + resp.content_type +
                     "\r\nContent-Length: " + std::to_string(resp.body.size()) +
                     "\r\nCache-Control: no-store\r\nConnection: " +
                     (keep_alive ? "keep-alive" : "close") +
                     "\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
  send_all(fd, head.data(), head.size());
  if (!resp.body.empty()) send_all(fd, resp.body.data(), resp.body.size());
  if (!keep_alive) net::shutdown(fd, net::kShutRdwr);
  return keep_alive;
}

}  // namespace crane
