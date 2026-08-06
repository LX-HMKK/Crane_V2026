// Cross-platform socket shim: maps the Berkeley-socket API used by the web
// server (http_server.cpp / web_ui.cpp) and the serial loopback test onto
// Winsock2 on Windows. POSIX call sites keep working unchanged because every
// wrapper normalizes Winsock errors into `errno` (MSVC aliases most WSA codes
// onto the same values, e.g. WSAEINTR=10004 vs EINTR=4 — the ones that differ
// are mapped explicitly in last_error()).
//
// Naming rule: the msg flags (kMsgNoSignal / kMsgDontWait) and kShutRdwr are
// DISTINCT constant names on purpose — on Linux MSG_NOSIGNAL / MSG_DONTWAIT /
// SHUT_RDWR are preprocessor macros, so `net::MSG_NOSIGNAL` would expand to
// `net::0` and fail to compile.
#pragma once

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef _WIN32
  // CMake already adds both globally for MSVC (NOMINMAX protects std::min/max
  // from windows.h); guard so the defines here are idempotent.
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <winsock2.h>  // must precede windows.h
  #include <ws2tcpip.h>
  #include <windows.h>  // FormatMessageA
#else
  #include <arpa/inet.h>
  #include <fcntl.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <sys/socket.h>
  #include <sys/time.h>  // timeval for SO_SNDTIMEO / SO_RCVTIMEO
  #include <unistd.h>
#endif

namespace crane {
namespace net {

#ifdef _WIN32
using Socket = SOCKET;  // 64-bit handle; never truncate into int
inline constexpr Socket kInvalidSocket = INVALID_SOCKET;
using ssize_t = std::ptrdiff_t;
inline constexpr int kMsgNoSignal = 0;  // no SIGPIPE on Windows
inline constexpr int kMsgDontWait = 0;  // use set_nonblocking + flags 0
inline constexpr int kShutRdwr = SD_BOTH;
#else
using Socket = int;
inline constexpr Socket kInvalidSocket = -1;
using ssize_t = ::ssize_t;
inline constexpr int kMsgNoSignal = MSG_NOSIGNAL;
inline constexpr int kMsgDontWait = MSG_DONTWAIT;
inline constexpr int kShutRdwr = SHUT_RDWR;
#endif

inline void ensure_winsock() {
#ifdef _WIN32
  static bool ok = [] {
    WSADATA d;
    return WSAStartup(MAKEWORD(2, 2), &d) == 0;
  }();
  if (!ok) throw std::runtime_error("WSAStartup failed");
#endif
}

// RAII: instantiate once in main() and in test mains (no-op on POSIX).
class Wsa {
 public:
  Wsa() { ensure_winsock(); }
};

inline int last_error() {
#ifdef _WIN32
  int e = WSAGetLastError();
  // Map the WSA codes that differ from the CRT errno values so call sites
  // comparing against EINTR / EWOULDBLOCK keep working unchanged.
  switch (e) {
    case WSAEWOULDBLOCK: return EWOULDBLOCK;
    case WSAEINTR: return EINTR;
    case WSAEINPROGRESS: return EINPROGRESS;
    case WSAENOTCONN: return ENOTCONN;
    case WSAECONNRESET: return ECONNRESET;
    case WSAECONNABORTED: return ECONNABORTED;
    case WSAETIMEDOUT: return ETIMEDOUT;
    default: return e;  // MSVC aliases the rest onto the same errno values
  }
#else
  return errno;
#endif
}

inline std::string error_message(int e) {
#ifdef _WIN32
  char buf[256];
  DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM |
                               FORMAT_MESSAGE_IGNORE_INSERTS,
                           nullptr, static_cast<DWORD>(e), 0, buf,
                           sizeof(buf), nullptr);
  if (n > 0) {
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = '\0';
    return buf;
  }
  return "socket error " + std::to_string(e);
#else
  return std::strerror(e);
#endif
}

inline Socket socket(int af, int type, int proto) {
  ensure_winsock();
  Socket s = ::socket(af, type, proto);
  if (s == kInvalidSocket) errno = last_error();
  return s;
}

inline int close(Socket s) {
#ifdef _WIN32
  return ::closesocket(s);
#else
  return ::close(s);
#endif
}

inline int shutdown(Socket s, int how) {
  int rc = ::shutdown(s, how);
  if (rc != 0) errno = last_error();
  return rc;
}

inline int bind(Socket s, const sockaddr* addr, size_t addrlen) {
  int rc;
#ifdef _WIN32
  rc = ::bind(s, addr, static_cast<int>(addrlen));
#else
  rc = ::bind(s, addr, static_cast<socklen_t>(addrlen));
#endif
  if (rc != 0) errno = last_error();
  return rc;
}

inline int listen(Socket s, int backlog) {
  int rc = ::listen(s, backlog);
  if (rc != 0) errno = last_error();
  return rc;
}

inline int setsockopt(Socket s, int level, int opt, const void* val,
                      size_t vallen) {
  int rc;
#ifdef _WIN32
  rc = ::setsockopt(s, level, opt, static_cast<const char*>(val),
                    static_cast<int>(vallen));
#else
  rc = ::setsockopt(s, level, opt, val, static_cast<socklen_t>(vallen));
#endif
  if (rc != 0) errno = last_error();
  return rc;
}

inline Socket accept(Socket s, sockaddr* addr, int* addrlen) {
  Socket rc;
#ifdef _WIN32
  rc = ::accept(s, addr, addrlen);
#else
  socklen_t alen = addrlen ? static_cast<socklen_t>(*addrlen) : 0;
  rc = ::accept(s, addr, addrlen ? &alen : nullptr);
  if (addrlen) *addrlen = static_cast<int>(alen);
#endif
  if (rc == kInvalidSocket) errno = last_error();
  return rc;
}

inline ssize_t send(Socket s, const void* buf, size_t len, int flags) {
  int rc;
#ifdef _WIN32
  rc = ::send(s, static_cast<const char*>(buf), static_cast<int>(len), flags);
#else
  rc = ::send(s, buf, len, flags);
#endif
  if (rc < 0) errno = last_error();
  return static_cast<ssize_t>(rc);
}

inline ssize_t recv(Socket s, void* buf, size_t len, int flags) {
  int rc;
#ifdef _WIN32
  rc = ::recv(s, static_cast<char*>(buf), static_cast<int>(len), flags);
#else
  rc = ::recv(s, buf, len, flags);
#endif
  if (rc < 0) errno = last_error();
  return static_cast<ssize_t>(rc);
}

// Write all bytes, retrying on EINTR and short sends. Returns false on any
// real error — including SO_SNDTIMEO expiry, which surfaces as EAGAIN /
// ETIMEDOUT after the timeout on a blocking socket. A stalled client
// (background-tab MJPEG throttle, network partition) therefore cannot hold
// the caller in send() forever once a send timeout is set at accept time.
inline bool send_all(Socket s, const char* data, size_t n) {
  size_t sent = 0;
  while (sent < n) {
    ssize_t rc = send(s, data + sent, n - sent, kMsgNoSignal);
    if (rc < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    sent += static_cast<size_t>(rc);
  }
  return true;
}

// SO_SNDTIMEO / SO_RCVTIMEO take different value types per platform: a DWORD
// of MILLISECONDS on Winsock, a struct timeval of seconds on POSIX. Passing a
// timeval on Windows would make Winsock read the first 4 bytes (tv_sec) as a
// millisecond count — tv_sec=5 becomes a 5ms timeout, not 5s. Always set the
// send timeout through this helper.
inline int set_send_timeout(Socket s, int timeout_sec) {
#ifdef _WIN32
  int ms = timeout_sec * 1000;
  return setsockopt(s, SOL_SOCKET, SO_SNDTIMEO,
                    static_cast<const char*>(static_cast<const void*>(&ms)),
                    sizeof(ms));
#else
  timeval tv{};
  tv.tv_sec = timeout_sec;
  return setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
}

inline int inet_pton(int af, const char* src, void* dst) {
  // On Windows the SDK also provides a real global ::inet_pton (InetPtonA is
  // just a #define for it); calling the qualified global avoids resolving to
  // this wrapper and recursing.
  return ::inet_pton(af, src, dst);
}

inline int getsockname(Socket s, sockaddr* addr, int* addrlen) {
  int rc;
#ifdef _WIN32
  rc = ::getsockname(s, addr, addrlen);
#else
  socklen_t alen = addrlen ? static_cast<socklen_t>(*addrlen) : 0;
  rc = ::getsockname(s, addr, addrlen ? &alen : nullptr);
  if (addrlen) *addrlen = static_cast<int>(alen);
#endif
  if (rc != 0) errno = last_error();
  return rc;
}

inline int connect(Socket s, const sockaddr* addr, int addrlen) {
  int rc;
#ifdef _WIN32
  rc = ::connect(s, addr, addrlen);
#else
  rc = ::connect(s, addr, static_cast<socklen_t>(addrlen));
#endif
  if (rc != 0) errno = last_error();
  return rc;
}

inline int set_nonblocking(Socket s, bool on) {
#ifdef _WIN32
  u_long mode = on ? 1 : 0;
  return ioctlsocket(s, FIONBIO, &mode);
#else
  int flags = fcntl(s, F_GETFL, 0);
  if (flags < 0) return -1;
  return fcntl(s, F_SETFL, on ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK));
#endif
}

}  // namespace net
}  // namespace crane
