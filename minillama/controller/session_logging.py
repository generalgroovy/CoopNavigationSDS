"""Background session logging for program monitoring and conversation metrics."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import threading
import time
import tracemalloc
from typing import Any

from minillama.controller.events import ConversationStepEvent, ProgramSegmentEvent, StructuredEvent, SystemEvent


STOP = object()

LOG_PROFILE_OFF = "off"
LOG_PROFILE_STARTUP = "startup"
LOG_PROFILE_RUNTIME = "runtime"
LOG_PROFILE_FULL = "full"
VALID_LOG_PROFILES = {
    LOG_PROFILE_OFF,
    LOG_PROFILE_STARTUP,
    LOG_PROFILE_RUNTIME,
    LOG_PROFILE_FULL,
}


def _normalize_profile(profile: str | None) -> str:
    profile = (profile or LOG_PROFILE_FULL).strip().lower()
    return profile if profile in VALID_LOG_PROFILES else LOG_PROFILE_FULL


def resource_snapshot() -> dict[str, Any]:
    """Return lightweight resource usage information for the current process."""
    current_bytes = peak_bytes = 0
    if tracemalloc.is_tracing():
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()

    cpu = os.times()
    return {
        "process_cpu_sec": round(cpu.user + cpu.system, 6),
        "wall_process_sec": round(time.process_time(), 6),
        "memory_current_bytes": int(current_bytes),
        "memory_peak_bytes": int(peak_bytes),
        "thread_count": threading.active_count(),
    }


class SessionLogger:
    """Write structured session logs on a background thread."""

    def __init__(self, session_name: str, log_dir: str | Path = "logs", profile: str = LOG_PROFILE_FULL):
        self.profile = _normalize_profile(profile)
        self.enabled = self.profile != LOG_PROFILE_OFF
        self.capture_resources = self.profile == LOG_PROFILE_FULL
        self.session_name = session_name
        self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_label = f"{session_name}-{self.session_id}"
        self.log_dir = Path(log_dir)
        self.started_at = time.time()
        self._closed = False
        if not self.enabled:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / f"{self.session_label}.jsonl"
        self.summary_path = self.log_dir / f"{self.session_label}.log"
        self.summary_json_path = self.log_dir / f"{self.session_label}-summary.json"
        self.queue: queue.Queue[Any] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        if self.capture_resources and not tracemalloc.is_tracing():
            tracemalloc.start()
        self._thread.start()
        self.submit(
            SystemEvent(
                self.session_label,
                self.started_at,
                "session.start",
                payload={
                    "session_name": self.session_name,
                    "log_dir": str(self.log_dir),
                    "profile": self.profile,
                },
            )
        )

    def submit(self, event: StructuredEvent):
        """Submit a structured event to the background writer."""
        if not self.enabled or self._closed:
            return
        self.queue.put(event)

    @contextmanager
    def segment(self, name: str, **payload):
        """Log a program segment with start/end timestamps and resource usage."""
        if not self.enabled or not self._should_log_segment(name):
            yield
            return

        started_at = time.time()
        started_perf = time.perf_counter()
        started_resource = resource_snapshot() if self.capture_resources else None
        self.submit(
            ProgramSegmentEvent(
                self.session_label,
                started_at,
                f"{name}.start",
                payload={
                    "segment": name,
                    "phase": "start",
                    **payload,
                    **({"resources": started_resource} if started_resource is not None else {}),
                },
            )
        )
        status = "ok"
        error = None
        try:
            yield
        except Exception as exc:
            status = "error"
            error = repr(exc)
            raise
        finally:
            ended_at = time.time()
            ended_resource = resource_snapshot() if self.capture_resources else None
            self.submit(
                ProgramSegmentEvent(
                    self.session_label,
                    ended_at,
                    f"{name}.end",
                    payload={
                        "segment": name,
                        "phase": "end",
                        "status": status,
                        "error": error,
                        "duration_sec": round(time.perf_counter() - started_perf, 6),
                        **({"resources_start": started_resource} if started_resource is not None else {}),
                        **({"resources_end": ended_resource} if ended_resource is not None else {}),
                        **payload,
                    },
                )
            )

    def log_step(self, *, turn: int, speaker: str, utterance: str, metrics: dict[str, Any]):
        """Record a single conversation step with metrics and resource usage."""
        if not self.enabled or not self._should_log_runtime():
            return
        self.submit(
            ConversationStepEvent(
                self.session_label,
                time.time(),
                f"turn.{turn}.{speaker.lower()}",
                payload={
                    "turn": turn,
                    "speaker": speaker,
                    "utterance": utterance,
                    "metrics": metrics,
                    **({"resources": resource_snapshot()} if self.capture_resources else {}),
                },
            )
        )

    def close(self):
        """Flush and stop the background writer."""
        if not self.enabled or self._closed:
            return
        self.submit(
            SystemEvent(
                self.session_label,
                time.time(),
                "session.end",
                payload={
                    "session_name": self.session_name,
                    "profile": self.profile,
                    **({"resources": resource_snapshot()} if self.capture_resources else {}),
                },
            )
        )
        self._closed = True
        self.queue.put(STOP)
        self._thread.join(timeout=10)

    def _run(self):
        event_count = 0
        with self.jsonl_path.open("w", encoding="utf-8") as jsonl_handle, self.summary_path.open(
            "w", encoding="utf-8"
        ) as summary_handle:
            while True:
                event = self.queue.get()
                if event is STOP:
                    break

                event_dict = event.to_dict() if hasattr(event, "to_dict") else asdict(event)
                jsonl_handle.write(json.dumps(event_dict, ensure_ascii=True) + "\n")
                jsonl_handle.flush()
                event_count += 1

                summary_line = self._format_summary(event_dict)
                if summary_line:
                    summary_handle.write(summary_line + "\n")
                    summary_handle.flush()

        self.summary_json_path.write_text(
            json.dumps(
                {
                    "session_label": self.session_label,
                    "started_at": self.started_at,
                    "events": event_count,
                    "jsonl_path": str(self.jsonl_path),
                    "summary_path": str(self.summary_path),
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _format_summary(event: dict[str, Any]) -> str | None:
        kind = event.get("kind")
        timestamp = datetime.fromtimestamp(event.get("timestamp", time.time())).strftime("%H:%M:%S")
        name = event.get("name", "")
        payload = event.get("payload", {})

        if kind == "program.segment":
            duration = payload.get("duration_sec")
            status = payload.get("status", "ok")
            return f"[{timestamp}] SEGMENT {name} status={status} duration={duration}s"
        if kind == "conversation.step":
            turn = payload.get("turn")
            speaker = payload.get("speaker")
            metrics = payload.get("metrics", {})
            route = metrics.get("route", "-")
            route_valid = metrics.get("route_valid", "-")
            route_goal = metrics.get("route_reaches_goal", "-")
            return (
                f"[{timestamp}] STEP turn={turn} speaker={speaker} route_valid={route_valid} "
                f"route_goal={route_goal} route={route}"
            )
        if kind == "system":
            if name == "session.start":
                return f"[{timestamp}] SESSION START {payload.get('session_name', '')}"
            if name == "session.end":
                return f"[{timestamp}] SESSION END {payload.get('session_name', '')}"
            return f"[{timestamp}] SYSTEM {name}"
        return None

    def _should_log_segment(self, name: str) -> bool:
        if self.profile == LOG_PROFILE_FULL:
            return True
        if self.profile == LOG_PROFILE_STARTUP:
            return name.startswith("model.load")
        if self.profile == LOG_PROFILE_RUNTIME:
            return name.startswith("dialog.run")
        return False

    def _should_log_runtime(self) -> bool:
        return self.profile in {LOG_PROFILE_RUNTIME, LOG_PROFILE_FULL}

    def accepts_tuple_event(self, event) -> bool:
        if not self.enabled or not isinstance(event, tuple) or not event:
            return False
        if self.profile == LOG_PROFILE_FULL:
            return True
        if self.profile == LOG_PROFILE_STARTUP:
            return False
        kind = event[0]
        return kind in {"message", "system", "warning", "route", "candidate", "telemetry", "metrics", "done"}


class MonitoringEventQueue:
    """Forward UI-compatible tuple events to a UI queue and structured logger."""

    def __init__(self, ui_queue, logger: SessionLogger):
        self.ui_queue = ui_queue
        self.logger = logger

    def put(self, event):
        self.ui_queue.put(event)
        if self.logger is None or not self.logger.accepts_tuple_event(event):
            return
        structured = self._to_structured_event(event)
        if structured is not None:
            self.logger.submit(structured)

    def emit_structured(self, event: StructuredEvent):
        self.logger.submit(event)

    def segment(self, name: str, **payload):
        if self.logger is None:
            return nullcontext()
        return self.logger.segment(name, **payload)

    def log_step(self, *, turn: int, speaker: str, utterance: str, metrics: dict[str, Any]):
        if self.logger is not None:
            self.logger.log_step(turn=turn, speaker=speaker, utterance=utterance, metrics=metrics)

    def close(self):
        if self.logger is not None:
            self.logger.close()

    @staticmethod
    def _to_structured_event(event):
        if not isinstance(event, tuple) or not event:
            return None

        kind = event[0]
        timestamp = time.time()
        if kind == "message" and len(event) >= 3:
            _, speaker, message = event
            return ConversationStepEvent(
                "ui-forward",
                timestamp,
                f"message.{speaker.lower()}",
                payload={"speaker": speaker, "utterance": message},
            )
        if kind == "system" and len(event) >= 2:
            return SystemEvent("ui-forward", timestamp, "system", payload={"message": event[1]})
        if kind == "warning" and len(event) >= 2:
            return SystemEvent("ui-forward", timestamp, "warning", payload={"message": event[1]})
        if kind == "route" and len(event) >= 2:
            return SystemEvent("ui-forward", timestamp, "route", payload={"route": event[1]})
        if kind == "candidate" and len(event) >= 2:
            return SystemEvent("ui-forward", timestamp, "candidate", payload={"candidate": event[1]})
        if kind == "telemetry" and len(event) >= 3:
            _, telemetry_type, payload = event
            return SystemEvent(
                "ui-forward",
                timestamp,
                f"telemetry.{telemetry_type}",
                payload=payload if isinstance(payload, dict) else {"value": payload},
            )
        if kind == "metrics" and len(event) >= 2:
            return SystemEvent("ui-forward", timestamp, "metrics", payload={"metrics": event[1]})
        if kind == "done":
            return SystemEvent("ui-forward", timestamp, "done")
        return None
