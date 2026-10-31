"""DIAG ONLY (not for submission): timestamps of setup phases and Triton compilations."""
import atexit
import sys
import threading
import time

T0 = time.perf_counter()
EVENTS = []


def mark(msg):
    EVENTS.append((round(time.perf_counter() - T0, 2), threading.current_thread().name[:8], msg))


def _report():
    print("[diag] " + " | ".join(f"{t}s {th} {m}" for t, th, m in EVENTS), file=sys.stderr)


def install():
    try:
        from triton.runtime.jit import JITFunction

        original = JITFunction._do_compile

        def _do_compile(self, *args, **kwargs):
            start = time.perf_counter()
            result = original(self, *args, **kwargs)
            mark(f"compile {getattr(self.fn, '__name__', '?')} {time.perf_counter() - start:.2f}s")
            return result

        JITFunction._do_compile = _do_compile
    except Exception as exc:  # noqa: BLE001
        mark(f"no compile hook {exc!r}")
    atexit.register(_report)
