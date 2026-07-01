"""Weak Solver: Local Ollama qwen2.5:3b for rollout evaluation.

Auto-starts the Ollama server if not already running.
"""

import json
import subprocess
import sys
import threading
import time
import urllib.request
from data_generation.config import (
    WEAK_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT,
    OLLAMA_START_TIMEOUT, OLLAMA_EXECUTABLE,
)

# Module-level state for once-per-process server launch
_server_started = False
_start_lock = threading.Lock()


def _check_server_alive(base_url=OLLAMA_BASE_URL):
    """Return True if Ollama API is responding."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def _start_ollama_serve():
    """Launch ollama serve as a detached background process (Windows)."""
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = (
            subprocess.CREATE_NO_WINDOW
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
        )

    subprocess.Popen(
        [OLLAMA_EXECUTABLE, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )


def ensure_server_running(timeout=OLLAMA_START_TIMEOUT):
    """Check if Ollama server is alive; start it if not.

    Thread-safe: only one caller launches the server process.
    Returns True if server is alive, False if timed out.
    """
    global _server_started

    # Fast path: already confirmed running this session
    if _server_started:
        return True

    with _start_lock:
        # Double-check inside lock
        if _server_started:
            return True

        if _check_server_alive():
            _server_started = True
            return True

        # Server not responding — launch it
        _start_ollama_serve()

        # Poll with exponential-backoff capped at 2s
        deadline = time.time() + timeout
        interval = 0.5
        while time.time() < deadline:
            time.sleep(interval)
            if _check_server_alive():
                _server_started = True
                return True
            interval = min(interval * 1.3, 2.0)

        return False


class WeakSolver:
    """Sends queries to local Ollama instance, returns text response.

    On first instantiation, auto-starts the Ollama server if needed.
    """

    def __init__(self, model=None, base_url=None):
        alive = ensure_server_running()
        if not alive:
            raise RuntimeError(
                f"Ollama server not reachable at {base_url or OLLAMA_BASE_URL} "
                f"after {OLLAMA_START_TIMEOUT}s. Check that Ollama is installed "
                f"and {OLLAMA_EXECUTABLE} exists."
            )

        self.model = model or WEAK_MODEL
        self.base_url = base_url or OLLAMA_BASE_URL

    def solve(self, query, context_chunks=None, temperature=0.0):
        """Send query and optional context to Ollama, return response text.

        Args:
            query: The question text
            context_chunks: Optional list of relevant source texts
            temperature: Generation temperature (0.0 for deterministic)

        Returns:
            str: Model response text
        """
        prompt = query
        if context_chunks:
            ctx = "\n\n".join(context_chunks[:5])
            prompt = f"参考以下标准条文：\n\n{ctx}\n\n问题：{query}\n\n请基于参考条文回答问题。"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 2048},
        }

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "")
        except Exception as e:
            return f"[WeakSolver Error: {e}]"

    def rollout(self, query, context_chunks=None, n=3):
        """Run n independent rollouts and return list of responses.

        Used by LoopJudge to evaluate rollout variance for GRPO suitability.
        """
        return [self.solve(query, context_chunks) for _ in range(n)]
