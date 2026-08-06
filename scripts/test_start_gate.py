#!/usr/bin/env python3
"""Integration test: START-gated recognition + DEBUG override (pty UART).

Runs crane_vision with a pseudo-terminal standing in for the serial UART and
verifies the recognition gate end to end:

  Run A (mock cameras): gate opens on START, closes when START drops, DEBUG
      forces recognition. Signalled by inference fps (5s rate window) and by
      task-log activity (mock scenes are logic-rejected, so task lines only
      appear while recognition actually runs).
  Run B (static images): a valid 12-byte result is produced only under START,
      stays sticky after START drops.

Usage (inside the WSL build env):  python3 scripts/test_start_gate.py
"""
import json
import os
import pty
import subprocess
import sys
import threading
import time
import urllib.request

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(PROJ, "build/crane_vision")
CFG = os.path.join(PROJ, "configs/vision.yaml")
TMP_CFG = "/tmp/crane_start_gate_cfg.yaml"
LOG = "/tmp/crane_start_gate.log"
PORT = 8099

failures = []


def http(path):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (PORT, path),
                                timeout=5) as r:
        return json.load(r)


def post(path):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.load(r)


def read_log():
    if not os.path.exists(LOG):
        return ""
    with open(LOG, "rb") as f:
        return f.read().decode("utf-8", "replace")


def task_lines():
    lg = read_log()
    return sum(lg.count(m) for m in ("Result updated",
                                     "Result generation failed",
                                     "No valid result"))


def check(name, cond, detail=""):
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, detail))
    if not cond:
        failures.append(name)


def fps(s):
    return s["bean_fps"] + s["digit_fps"]


def launched(extra_args):
    if os.path.exists(LOG):
        os.remove(LOG)
    master, slave = pty.openpty()
    slave_path = os.ttyname(slave)
    with open(CFG) as f:
        content = f.read()
    content = content.replace("port: /dev/ttyS1", "port: %s" % slave_path)
    with open(TMP_CFG, "w") as f:
        f.write(content)
    logf = open(LOG, "wb")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "%s/build:%s/tools/onnxruntime/lib" % (PROJ, PROJ)
    app = subprocess.Popen(
        [BIN, "--config", TMP_CFG, *extra_args, "--web-port", str(PORT)],
        stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=PROJ)
    for _ in range(120):
        try:
            http("/api/status")
            return master, app, logf
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("app did not start; log tail:\n" + read_log()[-2000:])


def teardown(master, app, logf):
    try:
        post("/api/exit")
    except Exception:
        pass
    try:
        app.wait(timeout=10)
    except subprocess.TimeoutExpired:
        app.kill()
        app.wait()
    try:
        os.close(master)
    except Exception:
        pass
    logf.close()


def send_start(master, seconds, interval=0.1):
    deadline = time.time() + seconds
    while time.time() < deadline:
        os.write(master, bytes([0xA5, 0x5A, 0x01]))
        time.sleep(interval)


def run_a():
    print("== Run A: mock cameras — gate open/close + DEBUG ==")
    master, app, logf = launched(["--mock-cameras"])
    try:
        # no START yet -> recognition gated off
        time.sleep(1.0)
        s = http("/api/status")
        check("A.no-START start off", s["start"] is False, s)
        check("A.no-START gate closed", s["recognition"] is False, s)
        base = task_lines()
        time.sleep(2.0)
        check("A.no-START no task activity", task_lines() == base,
              "base=%d now=%d" % (base, task_lines()))

        # START on for ~3s -> recognition runs
        t = threading.Thread(target=send_start, args=(master, 3.0))
        t.start()
        time.sleep(1.0)
        s = http("/api/status")
        check("A.START flag on while sending", s["start"] is True, s)
        check("A.START gate open", s["recognition"] is True, s)
        t.join()
        time.sleep(2.0)
        check("A.START task activity (inference ran)",
              task_lines() > base,
              "base=%d now=%d" % (base, task_lines()))

        # START drops -> recognition stops
        time.sleep(1.5)  # past start_timeout_sec=0.3
        s = http("/api/status")
        check("A.START-off flag off", s["start"] is False, s)
        check("A.START-off gate closed", s["recognition"] is False, s)
        off_base = task_lines()
        time.sleep(1.5)
        check("A.START-off no new task activity", task_lines() == off_base,
              "base=%d now=%d" % (off_base, task_lines()))

        # DEBUG on -> forced recognition despite no START
        post("/api/debug?mode=1")
        time.sleep(0.5)
        s = http("/api/status")
        check("A.DEBUG-on debug flag", s["debug"] is True, s)
        check("A.DEBUG-on gate forced open", s["recognition"] is True, s)
        check("A.DEBUG-on start still off", s["start"] is False, s)
        time.sleep(6.0)  # full rate window with DEBUG on -> fps > 0
        s = http("/api/status")
        check("A.DEBUG-on inference ran (fps > 0)", fps(s) > 0, s)

        # DEBUG off -> gated off again
        post("/api/debug?mode=0")
        time.sleep(0.5)
        s = http("/api/status")
        check("A.DEBUG-off debug flag", s["debug"] is False, s)
        check("A.DEBUG-off gate closed", s["recognition"] is False, s)
        off2_base = task_lines()
        time.sleep(1.5)
        check("A.DEBUG-off no new task activity", task_lines() == off2_base,
              "base=%d now=%d" % (off2_base, task_lines()))
    finally:
        teardown(master, app, logf)


def run_b():
    print("== Run B: static images — valid result + sticky under START ==")
    master, app, logf = launched(
        ["--bean-image", "tests/data/beans_2.jpg",
         "--digit-image", "tests/data/num_2.jpg"])
    try:
        time.sleep(1.5)
        base = task_lines()
        s = http("/api/status")
        check("B.no-START no valid result", s["recognition"] is False and
              base == 0 and s["frame"] is None, s)

        t = threading.Thread(target=send_start, args=(master, 3.0))
        t.start()
        time.sleep(1.0)
        s = http("/api/status")
        check("B.START flag on", s["start"] is True, s)
        check("B.START gate open", s["recognition"] is True, s)
        t.join()
        time.sleep(2.5)  # inference + task combine settle
        s = http("/api/status")
        check("B.START valid result produced",
              task_lines() > base and s["frame"] is not None, s)
        print("  frame =", s["frame"])

        time.sleep(2.0)  # past start_timeout_sec=0.3
        s = http("/api/status")
        check("B.START-off flag off", s["start"] is False, s)
        check("B.START-off gate closed", s["recognition"] is False, s)
        check("B.START-off frame sticky", s["frame"] is not None, s)
    finally:
        teardown(master, app, logf)


def main():
    run_a()
    run_b()
    print("\n--- app log tail ---")
    print(read_log()[-1200:])
    if failures:
        print("\nRESULT: %d FAILURE(S): %s" % (len(failures), failures))
        return 1
    print("\nRESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
