"""Competition result coordination outside the Tkinter UI layer."""

import logging
import threading
import time
from dataclasses import dataclass


RESULT_FAILURE_LOG_INTERVAL_SEC = 5.0
RESULT_UPDATE_LOG_INTERVAL_SEC = 1.0


@dataclass(frozen=True)
class CompetitionResult:
    """One completed logical result for UI overlay rendering."""

    bean_zones: dict
    digit_zones: dict


class CompetitionResultController:
    """Pair camera results, generate protocol frames, and update serial output.

    This class owns protocol publication state and has no Tkinter or camera
    dependency. Callers provide one coherent pair of detections plus their
    capture timestamps and retain responsibility for rendering its result.
    """

    def __init__(
        self,
        task_generator,
        serial,
        logic_cfg,
        *,
        update_interval,
        pair_max_skew_sec,
    ):
        self.task_generator = task_generator
        self.serial = serial
        self.logic_cfg = logic_cfg
        self.update_interval = float(update_interval)
        self.pair_max_skew_sec = float(pair_max_skew_sec)

        self._lock = threading.Lock()
        self._last_frame = None
        self._last_updated_at = None
        self._consecutive_failures = 0
        self._last_failure_log_at = None
        self._last_result_log_at = None

    @property
    def consecutive_failures(self):
        with self._lock:
            return self._consecutive_failures

    @staticmethod
    def _captures_are_compatible(bean_meta, digit_meta, max_skew_sec):
        if bean_meta is None or digit_meta is None:
            return True
        bean_at = bean_meta.get("captured_at")
        digit_at = digit_meta.get("captured_at")
        if bean_at is None or digit_at is None:
            return True
        return abs(bean_at - digit_at) <= max_skew_sec

    def clear(self):
        """Clear cached output while the START gate is inactive."""
        with self._lock:
            dirty = self._last_frame is not None or self._last_updated_at is not None
            dirty = dirty or self._consecutive_failures != 0
            self._last_frame = None
            self._last_updated_at = None
            self._consecutive_failures = 0
            if self.serial is not None and self.serial.current_frame is not None:
                self.serial.current_frame = None
                dirty = True
            return dirty

    def update(
        self,
        bean_dets,
        digit_dets,
        *,
        bean_meta=None,
        digit_meta=None,
        cameras_healthy=True,
        debug_forced=False,
    ):
        """Process one latest-result snapshot and return overlay data if valid."""
        if self.task_generator is None or bean_dets is None or digit_dets is None:
            return None
        if not cameras_healthy:
            return None
        if not self._captures_are_compatible(
            bean_meta, digit_meta, self.pair_max_skew_sec
        ):
            return None

        with self._lock:
            now = time.monotonic()
            if (
                self._last_updated_at is not None
                and now - self._last_updated_at < self.update_interval
            ):
                return None
            self._last_updated_at = now

            frame, bean_zones, digit_zones, reason = (
                self.task_generator.generate_frame_with_completion(
                    bean_dets,
                    digit_dets,
                    confidence_threshold=self.logic_cfg.get("confidence_threshold", 0.5),
                    max_inferred_beans=self.logic_cfg.get("max_inferred_beans", 1),
                    max_inferred_digits=self.logic_cfg.get("max_inferred_digits", 2),
                )
            )
            if frame is not None:
                result = CompetitionResult(bean_zones, digit_zones)
                if debug_forced:
                    logging.debug("DEBUG inference produced a result (display-only)")
                    return result

                self._consecutive_failures = 0
                self._last_failure_log_at = None
                if frame != self._last_frame:
                    if self.serial is not None:
                        self.serial.current_frame = frame
                    self._last_frame = frame
                    inferred = sum(
                        1
                        for result_zone in (bean_zones | digit_zones).values()
                        if result_zone[1] == "inferred"
                    )
                    self._log_result_update(frame, inferred, now)
                return result

            if debug_forced:
                logging.warning("DEBUG inference produced no result: %s", reason)
                return None

            self._consecutive_failures += 1
            if self._consecutive_failures >= 10:
                if self.serial is not None:
                    self.serial.current_frame = None
                self._log_result_failure(reason, now)
            else:
                logging.warning(
                    "generate_frame failed (%d/%d): %s",
                    self._consecutive_failures,
                    10,
                    reason,
                )
            return None

    def _log_result_failure(self, reason, now):
        if (
            self._last_failure_log_at is not None
            and now - self._last_failure_log_at < RESULT_FAILURE_LOG_INTERVAL_SEC
        ):
            return
        logging.error(
            "generate_frame failed %d times, clearing result: %s",
            self._consecutive_failures,
            reason,
        )
        self._last_failure_log_at = now

    def _log_result_update(self, frame, inferred, now):
        if (
            self._last_result_log_at is not None
            and now - self._last_result_log_at < RESULT_UPDATE_LOG_INTERVAL_SEC
        ):
            return
        logging.info(
            "Result updated: C1-C3=%s, N4-N8=%s (%d inferred)",
            frame[:3],
            frame[3:],
            inferred,
        )
        self._last_result_log_at = now
