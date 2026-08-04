#include "serial_port.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <set>
#include <stdexcept>
#include <system_error>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

#include "common/logging.hpp"

namespace crane {

namespace {

double monotonic_sec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void throw_errno(const char* what) {
  throw std::system_error(errno, std::generic_category(), what);
}

speed_t baud_to_speed(int baudrate) {
  switch (baudrate) {
#ifdef B921600
    case 921600:
      return B921600;
#endif
#ifdef B460800
    case 460800:
      return B460800;
#endif
    case 230400:
      return B230400;
    case 115200:
      return B115200;
    case 57600:
      return B57600;
    case 38400:
      return B38400;
    case 19200:
      return B19200;
    case 9600:
      return B9600;
    default:
      throw std::invalid_argument("unsupported baudrate " +
                                  std::to_string(baudrate));
  }
}

}  // namespace

// ── TermiosSerialIO ───────────────────────────────────────────────────────

TermiosSerialIO::TermiosSerialIO(const std::string& device, int baudrate,
                                 double timeout_sec)
    : timeout_sec_(timeout_sec) {
  fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0) throw_errno(("open " + device).c_str());

  termios tio{};
  if (tcgetattr(fd_, &tio) != 0) {
    int e = errno;
    ::close(fd_);
    errno = e;
    throw_errno("tcgetattr");
  }
  cfmakeraw(&tio);
  // 8/N/1, no flow control, receiver enabled, ignore modem lines.
  tio.c_cflag &= ~(CSIZE | PARENB | CSTOPB);
#ifdef CRTSCTS
  tio.c_cflag &= ~CRTSCTS;
#endif
  tio.c_cflag |= CS8 | CLOCAL | CREAD;
  tio.c_iflag &= ~(IXON | IXOFF | IXANY);
  tio.c_cc[VMIN] = 0;
  tio.c_cc[VTIME] = 0;

  speed_t speed = baud_to_speed(baudrate);
  cfsetispeed(&tio, speed);
  cfsetospeed(&tio, speed);
  if (tcsetattr(fd_, TCSANOW, &tio) != 0) {
    int e = errno;
    ::close(fd_);
    errno = e;
    throw_errno("tcsetattr");
  }
  tcflush(fd_, TCIOFLUSH);
}

std::vector<uint8_t> TermiosSerialIO::read_available() {
  if (fd_ < 0) throw std::system_error(EBADF, std::generic_category(),
                                       "serial read on closed port");
  // pyserial read(in_waiting): fetch whatever the driver already holds.
  int waiting = 0;
  if (::ioctl(fd_, FIONREAD, &waiting) != 0) throw_errno("ioctl FIONREAD");
  if (waiting <= 0) return {};

  std::vector<uint8_t> out(static_cast<size_t>(waiting));
  // The fd is non-blocking; read until EAGAIN or buffer filled.
  size_t got = 0;
  while (got < out.size()) {
    ssize_t n = ::read(fd_, out.data() + got, out.size() - got);
    if (n > 0) {
      got += static_cast<size_t>(n);
    } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      break;
    } else if (n < 0 && errno == EINTR) {
      continue;
    } else if (n < 0) {
      throw_errno("serial read");
    } else {
      break;  // EOF
    }
  }
  out.resize(got);
  return out;
}

size_t TermiosSerialIO::write_some(const uint8_t* data, size_t n) {
  if (fd_ < 0) throw std::system_error(EBADF, std::generic_category(),
                                       "serial write on closed port");
  // Bound the write like pyserial's write_timeout.
  fd_set wfds;
  FD_ZERO(&wfds);
  FD_SET(fd_, &wfds);
  timeval tv{};
  tv.tv_sec = static_cast<time_t>(timeout_sec_);
  tv.tv_usec = static_cast<suseconds_t>(
      (timeout_sec_ - static_cast<double>(tv.tv_sec)) * 1e6);
  int rc;
  do {
    rc = ::select(fd_ + 1, nullptr, &wfds, nullptr, &tv);
  } while (rc < 0 && errno == EINTR);
  if (rc < 0) throw_errno("serial write select");
  if (rc == 0) return 0;  // timed out: nothing enqueued

  ssize_t written = ::write(fd_, data, n);
  if (written < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
    throw_errno("serial write");
  }
  return static_cast<size_t>(written);
}

void TermiosSerialIO::close() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

std::unique_ptr<TermiosSerialIO> open_serial_io(const std::string& device,
                                                int baudrate,
                                                double timeout_sec) {
  return std::make_unique<TermiosSerialIO>(device, baudrate, timeout_sec);
}

// ── SerialPort ────────────────────────────────────────────────────────────

SerialPort::SerialPort(std::unique_ptr<ISerialIO> io) : io_(std::move(io)) {}

bool SerialPort::poll_start() {
  std::vector<uint8_t> data = io_->read_available();
  if (!data.empty()) {
    buf_.insert(buf_.end(), data.begin(), data.end());
  }

  auto it = std::find_end(buf_.begin(), buf_.end(), kStartCommand.begin(),
                          kStartCommand.end());
  if (it != buf_.end()) {
    last_start_time_.store(monotonic_sec(), std::memory_order_relaxed);
    buf_.erase(buf_.begin(), it + kStartCommand.size());
    constexpr size_t max_keep = kStartCommand.size() - 1;
    if (buf_.size() > max_keep) {
      buf_.erase(buf_.begin(), buf_.end() - max_keep);
    }
    return true;
  }

  constexpr size_t max_keep = kStartCommand.size() - 1;
  if (buf_.size() > max_keep) {
    buf_.erase(buf_.begin(), buf_.end() - max_keep);
  }
  return false;
}

bool SerialPort::is_start_active(double timeout) const {
  return monotonic_sec() - last_start_time_.load(std::memory_order_relaxed) <
         timeout;
}

bool SerialPort::wait_start(double timeout) {
  double deadline = timeout >= 0 ? monotonic_sec() + timeout : -1.0;
  while (true) {
    if (deadline >= 0 && monotonic_sec() > deadline) return false;
    poll_start();
    if (is_start_active()) return true;
  }
}

std::array<uint8_t, kResultFrameSize> SerialPort::encode_frame(
    const FramePayload& payload) {
  std::array<uint8_t, 3> colors = {payload[0], payload[1], payload[2]};
  std::array<uint8_t, 5> digits = {payload[3], payload[4], payload[5],
                                   payload[6], payload[7]};
  {
    std::set<int> c(colors.begin(), colors.end());
    if (c != std::set<int>{1, 2, 3}) {
      throw std::invalid_argument(
          "Pickup colors C1-C3 must be 1,2,3 exactly once");
    }
    std::set<int> d(digits.begin(), digits.end());
    if (d != std::set<int>{1, 2, 3, 4, 5}) {
      throw std::invalid_argument(
          "Placement digits N4-N8 must be 1..5 exactly once");
    }
  }
  std::array<uint8_t, kResultFrameSize> frame = {};
  frame[0] = kResultHead[0];
  frame[1] = kResultHead[1];
  std::copy(payload.begin(), payload.end(), frame.begin() + 2);
  frame[10] = kResultTail[0];
  frame[11] = kResultTail[1];
  return frame;
}

void SerialPort::send_result(const FramePayload& payload) {
  auto frame = encode_frame(payload);
  size_t written = io_->write_some(frame.data(), frame.size());
  if (written != frame.size()) {
    throw std::runtime_error("Serial write timed out after " +
                             std::to_string(written) + "/" +
                             std::to_string(frame.size()) + " result bytes");
  }
  // No flush(): write_timeout already bounds the kernel enqueue; a driver
  // drain can block indefinitely despite a finite write timeout.
}

void SerialPort::close() { io_->close(); }

// ── SerialThread ────────────────────────────────────────────────────────────

SerialThread::SerialThread(SerialIOFactory io_factory, double start_timeout,
                           double send_interval, bool reconnect_enabled,
                           int reconnect_after, double reconnect_interval)
    : io_factory_(std::move(io_factory)),
      start_timeout_(start_timeout),
      send_interval_(send_interval),
      reconnect_enabled_(reconnect_enabled),
      reconnect_after_(std::max(1, reconnect_after)),
      reconnect_interval_(std::max(0.05, reconnect_interval)) {
  port_ = std::make_shared<SerialPort>(io_factory_());
}

std::unique_ptr<SerialThread> SerialThread::for_device(
    const std::string& device, int baudrate, double timeout,
    double start_timeout, double send_interval, bool reconnect_enabled,
    int reconnect_after, double reconnect_interval) {
  SerialIOFactory factory = [device, baudrate, timeout]() {
    return std::unique_ptr<ISerialIO>(
        open_serial_io(device, baudrate, timeout).release());
  };
  return std::make_unique<SerialThread>(std::move(factory), start_timeout,
                                        send_interval, reconnect_enabled,
                                        reconnect_after, reconnect_interval);
}

void SerialThread::set_current_frame(std::optional<FramePayload> frame) {
  std::lock_guard<std::mutex> lock(frame_mutex_);
  current_frame_ = frame;
}

std::optional<FramePayload> SerialThread::current_frame() const {
  std::lock_guard<std::mutex> lock(frame_mutex_);
  return current_frame_;
}

bool SerialThread::connected() const {
  std::lock_guard<std::mutex> lock(port_mutex_);
  return port_ != nullptr;
}

bool SerialThread::is_start_active() const {
  std::shared_ptr<SerialPort> port;
  {
    std::lock_guard<std::mutex> lock(port_mutex_);
    port = port_;
  }
  if (!port) return false;
  return port->is_start_active(start_timeout_);
}

void SerialThread::start() {
  if (thread_.joinable()) {
    throw std::runtime_error("SerialThread is already running");
  }
  running_ = true;
  thread_ = std::thread(&SerialThread::run, this);
}

void SerialThread::stop() {
  running_ = false;
  if (thread_.joinable()) thread_.join();
  std::shared_ptr<SerialPort> port;
  {
    std::lock_guard<std::mutex> lock(port_mutex_);
    port = port_;
  }
  if (port) port->close();
}

void SerialThread::run() {
  double last_send = 0.0;
  while (running_) {
    try {
      std::shared_ptr<SerialPort> port;
      {
        std::lock_guard<std::mutex> lock(port_mutex_);
        port = port_;
      }
      if (!port) {
        // Previous reopen failed; retry on the reconnect interval.
        maybe_reopen();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      port->poll_start();
      if (port->is_start_active(start_timeout_)) {
        auto frame = current_frame();
        if (frame.has_value()) {
          reported_missing_frame_ = false;
          double now = monotonic_sec();
          if (now - last_send >= send_interval_) {
            port->send_result(*frame);
            if (!reported_transmission_) {
              LOG_INFO("Serial result transmission active");
              reported_transmission_ = true;
            }
            last_send = now;
          }
        } else if (!reported_missing_frame_) {
          LOG_WARN("Serial START is active but no valid vision result is "
                   "available");
          reported_missing_frame_ = true;
        }
      } else {
        reported_missing_frame_ = false;
        reported_transmission_ = false;
      }
      consecutive_failures_ = 0;
    } catch (const std::exception& e) {
      ++consecutive_failures_;
      maybe_reopen();
      LOG_ERROR("SerialThread send failed: %s", e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));  // 200 Hz
  }
}

void SerialThread::maybe_reopen() {
  if (!reconnect_enabled_) return;
  if (consecutive_failures_ < reconnect_after_) return;
  if (monotonic_sec() - last_reopen_at_ < reconnect_interval_) return;
  last_reopen_at_ = monotonic_sec();
  reopen();
}

void SerialThread::reopen() {
  std::lock_guard<std::mutex> lock(port_mutex_);
  if (port_) {
    try {
      port_->close();
    } catch (...) {
      LOG_WARN("Failed to close serial port before reopen");
    }
  }
  port_.reset();
  try {
    port_ = std::make_shared<SerialPort>(io_factory_());
    LOG_INFO("Serial port reconnected");
    consecutive_failures_ = 0;
  } catch (const std::exception&) {
    LOG_WARN("Serial port reconnect failed");
    port_.reset();
  }
}

}  // namespace crane
