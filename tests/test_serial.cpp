// Serial protocol tests, ported from crane/tools/crane_tests/test_serial.py.
// Uses socketpair loopback instead of pyserial's loop:// URL.
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/socket.h>
#include <unistd.h>

#include "io/serial_port.hpp"

namespace {

int g_pass = 0, g_fail = 0;

#define CHECK(cond, ...)                              \
  do {                                                \
    if (cond) {                                       \
      ++g_pass;                                       \
    } else {                                          \
      ++g_fail;                                       \
      std::printf("FAIL %s:%d: ", __func__, __LINE__); \
      std::printf(__VA_ARGS__);                       \
      std::printf("\n");                              \
    }                                                 \
  } while (0)

// Loopback IO over one end of a socketpair; the test drives the other end.
class LoopbackIO : public crane::ISerialIO {
 public:
  explicit LoopbackIO(int fd) : fd_(fd) {}
  ~LoopbackIO() override { close(); }

  std::vector<uint8_t> read_available() override {
    std::vector<uint8_t> out(256);
    ssize_t n = ::recv(fd_, out.data(), out.size(), MSG_DONTWAIT);
    if (n <= 0) return {};
    out.resize(static_cast<size_t>(n));
    return out;
  }
  size_t write_some(const uint8_t* data, size_t n) override {
    ssize_t rc = ::send(fd_, data, n, MSG_NOSIGNAL);
    return rc < 0 ? 0 : static_cast<size_t>(rc);
  }
  void close() override {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

 private:
  int fd_;
};

struct Loop {
  int fds[2];
  std::unique_ptr<crane::SerialPort> port;

  Loop() {
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
      throw std::runtime_error("socketpair failed");
    }
    port = std::make_unique<crane::SerialPort>(
        std::make_unique<LoopbackIO>(fds[0]));
  }
  ~Loop() { ::close(fds[1]); }

  void write(const std::vector<uint8_t>& data) {
    ::send(fds[1], data.data(), data.size(), MSG_NOSIGNAL);
  }
  std::vector<uint8_t> read(size_t n, int timeout_ms = 500) {
    std::vector<uint8_t> out;
    out.reserve(n);
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);
    while (out.size() < n && std::chrono::steady_clock::now() < deadline) {
      uint8_t buf[64];
      ssize_t rc = ::recv(fds[1], buf, std::min(sizeof(buf), n - out.size()),
                          MSG_DONTWAIT);
      if (rc > 0) {
        out.insert(out.end(), buf, buf + rc);
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    }
    return out;
  }
};

const std::vector<uint8_t> kStart = {0xA5, 0x5A, 0x01};

void test_encode_frame_valid() {
  auto frame = crane::SerialPort::encode_frame({1, 2, 3, 3, 1, 5, 2, 4});
  std::array<uint8_t, 12> want = {0xA5, 0x5A, 1, 2, 3, 3, 1, 5, 2, 4, 0x0D,
                                  0x0A};
  CHECK(frame == want, "frame bytes mismatch");
}

void test_encode_frame_rejects_duplicate_colors() {
  bool threw = false;
  try {
    crane::SerialPort::encode_frame({1, 1, 2, 1, 2, 3, 4, 5});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw, "duplicate colors accepted");
}

void test_encode_frame_rejects_duplicate_digits() {
  bool threw = false;
  try {
    crane::SerialPort::encode_frame({1, 2, 3, 1, 2, 3, 4, 4});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw, "duplicate digits accepted");
}

void test_encode_frame_rejects_out_of_range() {
  bool threw = false;
  try {
    crane::SerialPort::encode_frame({0, 1, 2, 1, 2, 3, 4, 5});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw, "color 0 accepted");
}

void test_poll_start_detects_start_command() {
  Loop lp;
  lp.write({0x00, 0x01, 0xA5, 0x5A, 0x01, 0x00});
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  CHECK(lp.port->poll_start(), "START not detected");
  CHECK(lp.port->is_start_active(0.3), "START not active");
}

void test_poll_start_split_command() {
  Loop lp;
  lp.write({0xA5, 0x5A});
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  CHECK(!lp.port->poll_start(), "partial START falsely detected");
  lp.write({0x01});
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  CHECK(lp.port->poll_start(), "split START not detected");
}

void test_poll_start_discards_heartbeat_burst_keeps_split_prefix() {
  Loop lp;
  // burst of complete commands + trailing partial command prefix
  lp.write({0xA5, 0x5A, 0x01, 0xA5, 0x5A, 0x01, 0xA5, 0x5A});
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  CHECK(lp.port->poll_start(), "START not detected in burst");
  // The trailing A5 5A must survive so the command completes with the
  // next 0x01.
  lp.write({0x01});
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  CHECK(lp.port->poll_start(), "split START after burst not detected");
}

void test_is_start_active_times_out() {
  Loop lp;
  lp.write(kStart);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  CHECK(lp.port->poll_start(), "START not detected");
  CHECK(lp.port->is_start_active(0.05), "START should be active");
  std::this_thread::sleep_for(std::chrono::milliseconds(80));
  CHECK(!lp.port->is_start_active(0.05), "START should have timed out");
}

void test_wait_start_blocking() {
  Loop lp;
  std::thread writer([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    lp.write({0x00, 0x01, 0xA5, 0x5A, 0x01, 0x00});
  });
  bool ok = lp.port->wait_start(2.0);
  writer.join();
  CHECK(ok, "wait_start timed out");
}

void test_wait_start_timeout_returns_false() {
  Loop lp;
  CHECK(!lp.port->wait_start(0.1), "wait_start should time out");
}

void test_send_result_writes_correct_frame() {
  Loop lp;
  lp.port->send_result({1, 2, 3, 3, 1, 5, 2, 4});
  auto got = lp.read(12);
  std::vector<uint8_t> want = {0xA5, 0x5A, 1, 2, 3, 3, 1, 5, 2, 4, 0x0D,
                               0x0A};
  CHECK(got == want, "frame mismatch (got %zu bytes)", got.size());
}

void test_serial_thread_sends_at_interval_when_start_active() {
  int fds[2];
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
    CHECK(false, "socketpair failed");
    return;
  }
  crane::SerialIOFactory factory = [fd = fds[0]]() {
    return std::make_unique<LoopbackIO>(fd);
  };
  crane::SerialThread st(factory, /*start_timeout=*/0.3,
                         /*send_interval=*/0.05, /*reconnect_enabled=*/true,
                         /*reconnect_after=*/5, /*reconnect_interval=*/1.0);
  st.set_current_frame(crane::FramePayload{1, 2, 3, 1, 2, 3, 4, 5});
  st.start();
  // Drive START heartbeats from the other end.
  std::atomic<bool> drive{true};
  std::thread heartbeat([&] {
    while (drive) {
      ::send(fds[1], kStart.data(), kStart.size(), MSG_NOSIGNAL);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  drive = false;
  st.stop();
  heartbeat.join();

  auto got = [&] {
    std::vector<uint8_t> out;
    uint8_t buf[256];
    ssize_t n;
    while ((n = ::recv(fds[1], buf, sizeof(buf), MSG_DONTWAIT)) > 0) {
      out.insert(out.end(), buf, buf + n);
    }
    return out;
  }();
  ::close(fds[1]);

  // Expect roughly 400ms/50ms ≈ 8 frames (tolerance for scheduling).
  size_t frames = got.size() / 12;
  CHECK(frames >= 4 && frames <= 14, "expected ~8 frames, got %zu (%zu B)",
        frames, got.size());
  bool all_valid = got.size() % 12 == 0;
  for (size_t i = 0; i + 12 <= got.size() && all_valid; i += 12) {
    std::array<uint8_t, 12> f;
    std::memcpy(f.data(), got.data() + i, 12);
    std::array<uint8_t, 12> want = {0xA5, 0x5A, 1, 2, 3, 1, 2, 3, 4, 5, 0x0D,
                                    0x0A};
    all_valid = (f == want);
  }
  CHECK(all_valid, "frame content mismatch");
}

}  // namespace

int main() {
  test_encode_frame_valid();
  test_encode_frame_rejects_duplicate_colors();
  test_encode_frame_rejects_duplicate_digits();
  test_encode_frame_rejects_out_of_range();
  test_poll_start_detects_start_command();
  test_poll_start_split_command();
  test_poll_start_discards_heartbeat_burst_keeps_split_prefix();
  test_is_start_active_times_out();
  test_wait_start_blocking();
  test_wait_start_timeout_returns_false();
  test_send_result_writes_correct_frame();
  test_serial_thread_sends_at_interval_when_start_active();
  std::printf("%s: %d passed, %d failed\n", g_fail == 0 ? "PASS" : "FAIL",
              g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
