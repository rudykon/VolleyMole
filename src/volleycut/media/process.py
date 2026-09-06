"""Cancellable local subprocesses; always reap the child before returning."""
import os
import signal
import subprocess
import tempfile


def run_process(cmd, *, cancel_check=None, on_output=None):
    if cancel_check:
        cancel_check()
    with tempfile.TemporaryFile(mode="w+b") as stdout, tempfile.TemporaryFile(mode="w+b") as stderr:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                start_new_session=True)
        offset = 0
        try:
            while True:
                if cancel_check:
                    cancel_check()
                try:
                    proc.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if on_output:
                    chunk = os.pread(stdout.fileno(), 65536, offset)
                    offset += len(chunk)
                    if chunk:
                        on_output(chunk.decode("utf-8", errors="replace"))
            stdout.seek(0)
            stderr.seek(0)
            return subprocess.CompletedProcess(cmd, proc.returncode,
                                               stdout.read().decode("utf-8", errors="replace"),
                                               stderr.read().decode("utf-8", errors="replace"))
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
