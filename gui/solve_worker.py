"""
Background solve execution so the Tkinter main loop never blocks.

Only one solve is ever in flight; if a new request arrives while busy, it
replaces any pending request and is picked up as soon as the current solve
finishes ("latest request wins" coalescing) -- this is what makes debounced
live-re-solve safe without ever touching Tk widgets off the main thread.

Consumers must poll `poll()` from the Tk main thread (e.g. via
`root.after`) to receive results/errors/log lines.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional

from devices.layer import Contact
from devices.device import AlGaNDevice
from physics.self_consistent import SolverResult
from physics.coupled_solver import SolveCancelled


@dataclass
class SolveRequest:
    request_id: int
    layers: list
    contacts: List[Contact]
    T: float
    dx_nm: float
    include_spontaneous_polarization: bool
    quantum: bool
    n_states_e: int
    n_states_h: int
    max_iter: int
    tol: float
    alpha: float
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
    cancelled: bool = False


@dataclass
class SolveLog:
    """One line of the solver's verbose progress output, streamed live."""
    request_id: int
    text: str


class SolveWorker:
    def __init__(self) -> None:
        self._out: "queue.Queue[Any]" = queue.Queue()
        self._lock = threading.Lock()
        self._pending: Optional[SolveRequest] = None
        self._busy = False
        self._next_id = 0
        # Set by stop() to unwind the currently-running solve (checked from
        # the solve thread via cancel_check -- see physics.self_consistent's
        # outer loop and physics.coupled_solver's newton_krylov callback).
        # Cleared right before a new request actually starts running, so a
        # past Stop click can't immediately cancel the next solve.
        self._cancel_event = threading.Event()

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
                self._cancel_event.clear()
                threading.Thread(target=self._run, args=(req,), daemon=True).start()
            return req.request_id

    def stop(self) -> None:
        """Cancel whatever solve is currently running, and drop any request
        that was queued behind it (coalescing submit means at most one)."""
        with self._lock:
            self._pending = None
            self._cancel_event.set()

    def poll(self) -> List[Any]:
        """Drain and return all queued SolveDone/SolveLog messages."""
        msgs = []
        while True:
            try:
                msgs.append(self._out.get_nowait())
            except queue.Empty:
                break
        return msgs

    # ------------------------------------------------------------------
    def _run(self, req: SolveRequest) -> None:
        # A plain callback (not a sys.stdout redirect): solve_self_consistent
        # calls this directly wherever it would otherwise print(), so each
        # solve's log goes only to *this* request's queue entries -- safe to
        # run concurrently with anything else the process is doing on
        # stdout, unlike mutating the process-wide sys.stdout would be.
        def log_fn(text: str) -> None:
            self._out.put(SolveLog(request_id=req.request_id, text=text))

        try:
            device = AlGaNDevice(layers=req.layers, contacts=req.contacts,
                                  T=req.T, dx_nm=req.dx_nm,
                                  include_spontaneous_polarization=req.include_spontaneous_polarization)
            if req.sweep:
                results = device.sweep_voltage(
                    req.V_start, req.V_stop, n_steps=req.n_steps,
                    quantum=req.quantum,
                    alpha=req.alpha, max_iter=req.max_iter,
                    verbose=True, log_fn=log_fn,
                    cancel_check=self._cancel_event.is_set,
                )
                self._out.put(SolveDone(request_id=req.request_id, results=results))
            else:
                # solve_ramped (not a raw solve()) reaches a nonzero bias by
                # continuation from equilibrium, same as sweep_voltage does
                # per-step -- a direct one-shot jump to a large bias is what
                # made the coupled Newton-Krylov solve fail to converge at
                # high V_applied (see devices.device.AlGaNDevice.solve_ramped).
                result = device.solve_ramped(
                    V_applied=req.V_applied,
                    quantum=req.quantum, n_states_e=req.n_states_e,
                    n_states_h=req.n_states_h, max_iter=req.max_iter,
                    tol=req.tol, alpha=req.alpha,
                    verbose=True, log_fn=log_fn,
                    cancel_check=self._cancel_event.is_set,
                )
                self._out.put(SolveDone(request_id=req.request_id, result=result))
        except SolveCancelled:
            log_fn("Solve cancelled by user.")
            self._out.put(SolveDone(request_id=req.request_id, cancelled=True))
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
            self._cancel_event.clear()
        threading.Thread(target=self._run, args=(nxt,), daemon=True).start()
