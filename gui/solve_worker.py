"""
Background solve execution so the Tkinter main loop never blocks.

Only one solve is ever in flight; if a new request arrives while busy, it
replaces any pending request and is picked up as soon as the current solve
finishes ("latest request wins" coalescing) -- this is what makes debounced
live-re-solve safe without ever touching Tk widgets off the main thread.

Consumers must poll `poll()` from the Tk main thread (e.g. via
`root.after`) to receive results/errors.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional

from devices.layer import Contact
from devices.device import AlGaNDevice
from physics.self_consistent import SolverResult


@dataclass
class SolveRequest:
    request_id: int
    layers: list
    contacts: List[Contact]
    T: float
    dx_nm: float
    quantum: bool
    n_states_e: int
    n_states_h: int
    max_iter: int
    tol: float
    alpha: float
    R_series: float
    sweep: bool
    V_applied: float
    V_start: float
    V_stop: float
    n_steps: int


@dataclass
class SolveDone:
    request_id: int
    result: Optional[SolverResult] = None         # single-point solve
    results: Optional[List[SolverResult]] = None  # sweep
    error: Optional[str] = None


class SolveWorker:
    def __init__(self) -> None:
        self._out: "queue.Queue[Any]" = queue.Queue()
        self._lock = threading.Lock()
        self._pending: Optional[SolveRequest] = None
        self._busy = False
        self._next_id = 0

    def submit(self, req_kwargs: dict) -> int:
        """Coalescing submit: returns the request_id assigned to this call.
        Only the caller's own latest request_id (compared against messages
        drained by poll()) should be treated as authoritative by the GUI."""
        with self._lock:
            self._next_id += 1
            req = SolveRequest(request_id=self._next_id, **req_kwargs)
            if self._busy:
                self._pending = req
            else:
                self._busy = True
                threading.Thread(target=self._run, args=(req,), daemon=True).start()
            return req.request_id

    def poll(self) -> List[Any]:
        """Drain and return all queued SolveDone messages."""
        msgs = []
        while True:
            try:
                msgs.append(self._out.get_nowait())
            except queue.Empty:
                break
        return msgs

    # ------------------------------------------------------------------
    def _run(self, req: SolveRequest) -> None:
        try:
            device = AlGaNDevice(layers=req.layers, contacts=req.contacts,
                                  T=req.T, dx_nm=req.dx_nm)
            if req.sweep:
                results = device.sweep_voltage(
                    req.V_start, req.V_stop, n_steps=req.n_steps,
                    R_series=req.R_series, quantum=req.quantum,
                    alpha=req.alpha, max_iter=req.max_iter,
                )
                self._out.put(SolveDone(request_id=req.request_id, results=results))
            else:
                result = device.solve(
                    V_applied=req.V_applied, R_series=req.R_series,
                    quantum=req.quantum, n_states_e=req.n_states_e,
                    n_states_h=req.n_states_h, max_iter=req.max_iter,
                    tol=req.tol, alpha=req.alpha,
                )
                self._out.put(SolveDone(request_id=req.request_id, result=result))
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI, never crash
            self._out.put(SolveDone(request_id=req.request_id, error=str(exc)))
        finally:
            self._start_next()

    def _start_next(self) -> None:
        with self._lock:
            nxt = self._pending
            self._pending = None
            if nxt is None:
                self._busy = False
                return
        threading.Thread(target=self._run, args=(nxt,), daemon=True).start()
