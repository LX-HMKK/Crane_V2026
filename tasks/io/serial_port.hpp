#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "common/types.hpp"

namespace crane {

inline constexpr std::array<uint8_t, 3> kStartCommand = {0xA5, 0x5A, 0x01};
inline constexpr std::array<uint8_t, 2> kResultHead = {0xA5, 0x5A};
inline constexpr std::array<uint8_t, 2> kResultTail = {0x0D, 0x0A};
inline constexpr size_t kResultFrameSize = 12;

// ── low-level serial IO abstraction (testable without real UART) ──────
class ISerialIO {
 public:
  virtual ~ISerialIO() = default;
  // Read all currently-available bytes (empty when none), bounded by the
  // read timeout. Throws std::system_error on IO faults.
  virtual std::vector<uint8_t> read_available() = 0;
  // Write bytes, bounded by the write timeout. Returns bytes written
  // (short count = timeout). Throws std::system_error on IO faults.
  virtual size_t write_some(const uint8_t* data, size_t n) = 0;
  virtual void close() = 0;
};

using SerialIOFactory = std::function<std::unique_ptr<ISerialIO>()>;

// Real UART via termios: 8/N/1, no flow control, non-blocking + select.
class TermiosSerialIO : public ISerialIO {
 public:
  TermiosSerialIO(const std::string& device, int baudrate,
                  double timeout_sec = 0.05);
  ~TermiosSerialIO() override { close(); }

  std::vector<uint8_t> read_available() override;
  size_t write_some(const uint8_t* data, size_t n) override;
  void close() override;

 private:
  int fd_ = -1;
  double timeout_sec_;
};

// Open a UART and wrap it in an ISerialIO (throws on failure). On Windows
// this always throws — the dev-verification build runs with --no-serial, and
// the real COM-port backend lives only on the RDK X5 Linux board. Returns
// ISerialIO (not TermiosSerialIO) so SerialThread::for_device's temporary
// never forces the never-defined TermiosSerialIO::close() to be emitted on
// Windows.
std::unique_ptr<ISerialIO> open_serial_io(const std::string& device,
                                          int baudrate,
                                          double timeout_sec = 0.05);

// ── protocol layer (mirrors models/serial_port.py SerialPort) ──────────
class SerialPort {
 public:
  explicit SerialPort(std::unique_ptr<ISerialIO> io);

  // Read available RX bytes; returns true if a new START command was seen.
  // Buffer keeps at most len(START)-1 trailing bytes (split commands).
  bool poll_start();

  // True if a START command arrived within `timeout` seconds.
  bool is_start_active(double timeout = 0.3) const;

  // Blocking helper: poll until START arrives or timeout (seconds; <0 =
  // forever). For tests; the competition loop uses poll_start directly.
  bool wait_start(double timeout = -1.0);

  // Send the fixed 12-byte frame: A5 5A C1 C2 C3 N4 N5 N6 N7 N8 0D 0A.
  // Throws std::invalid_argument on illegal payload, std::system_error on
  // IO faults, std::runtime_error on short writes.
  void send_result(const FramePayload& frame);

  // Encode + validate only (exposed for tests).
  static std::array<uint8_t, kResultFrameSize> encode_frame(
      const FramePayload& payload);

  double last_start_time() const {
    return last_start_time_.load(std::memory_order_relaxed);
  }
  void close();

 private:
  std::unique_ptr<ISerialIO> io_;
  std::vector<uint8_t> buf_;
  // Written by the serial thread (poll_start), read by status threads.
  std::atomic<double> last_start_time_{0.0};  // monotonic seconds
};

// ── background serial thread (mirrors SerialThread) ─────────────────────
class SerialThread {
 public:
  SerialThread(SerialIOFactory io_factory, double start_timeout = 0.3,
               double send_interval = 0.1, bool reconnect_enabled = true,
               int reconnect_after = 5, double reconnect_interval = 1.0);
  ~SerialThread() { stop(); }

  // Convenience constructor for a real UART device.
  static std::unique_ptr<SerialThread> for_device(
      const std::string& device, int baudrate, double timeout = 0.05,
      double start_timeout = 0.3, double send_interval = 0.1,
      bool reconnect_enabled = true, int reconnect_after = 5,
      double reconnect_interval = 1.0);

  // Thread-safe current frame (nullopt = nothing valid to send).
  void set_current_frame(std::optional<FramePayload> frame);
  std::optional<FramePayload> current_frame() const;

  bool connected() const;
  bool is_start_active() const;
  // True while the RX loop is actively polling without error. During the 1s
  // error-backoff pause the thread cannot read START, so last_start_time_
  // ages out; the recognition gate treats an unhealthy RX loop as fail-open.
  bool rx_healthy() const {
    return rx_healthy_.load(std::memory_order_relaxed);
  }

  void start();
  void stop();

 private:
  void run();
  void maybe_reopen();
  void reopen();

  SerialIOFactory io_factory_;
  double start_timeout_;
  double send_interval_;
  bool reconnect_enabled_;
  int reconnect_after_;
  double reconnect_interval_;

  // port_ is shared between the serial thread (poll/send/reopen) and status
  // readers (connected/is_start_active). Take a copy under port_mutex_,
  // use it unlocked — reopen() swaps the pointer under the same mutex.
  //
  // Lock order invariant (checked by architecture review): the inference
  // workers in app.cpp may hold infer_mutex_ while acquiring port_mutex_ in
  // the recognition-gate predicate. Never acquire infer_mutex_ while holding
  // port_mutex_ — that would deadlock both inference workers.
  mutable std::mutex port_mutex_;
  std::shared_ptr<SerialPort> port_;
  int consecutive_failures_ = 0;
  double last_reopen_at_ = 0.0;

  mutable std::mutex frame_mutex_;
  std::optional<FramePayload> current_frame_;

  std::atomic<bool> running_{false};
  std::atomic<bool> rx_healthy_{true};
  std::thread thread_;
  bool reported_missing_frame_ = false;
  bool reported_transmission_ = false;
};

}  // namespace crane
