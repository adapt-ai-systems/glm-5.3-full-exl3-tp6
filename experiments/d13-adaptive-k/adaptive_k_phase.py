# SPDX-License-Identifier: Apache-2.0
"""Adaptive MTP draft length for GLM 5.3 Full TP6, optionally per decode phase.

Port of Matt Mastracci's adaptive-k scheduler (kindlingai/glm-5.3-flash-gx10, experimental/adaptive-k/adaptive_k.py,
Apache-2.0) from DFlash2 to the MTP drafter, plus decode phases.

Every step the scheduler picks k, the number of drafts the MTP head proposes this step (and the
next step verifies). MTP drafts sequentially, so a smaller k saves draft forwards as well as
verify tokens. k maximises expected tokens per ms, where expected tokens come from c[i], an EMA of
the rate draft i is accepted given draft i-1 was, and step ms from VLLM_ADAPTIVE_K_COST_MS
("verified tokens:ms,...", measured with "force N").

No num_speculative_tokens_per_batch_size: setting it makes e3-v2 force cudagraph_mode PIECEWISE.
Instead the scheduler writes scheduler_output.num_spec_tokens_to_schedule itself and the boot
captures FULL graphs at every bs*(k+1) (with D13/compilation.py not rounding sizes to k+1).

Phases (mode "phase"): each request is in reasoning (<think>..</think>), tool (<tool_call>..
</tool_call>), code (inside ``` fences in content) or content. Acceptance state is kept per
(request, phase); a request entering a phase it has not been in starts from that phase's global
prior (an EMA over all requests, seeded from D12 measurements). A phase change switches k on the
next schedule without the hysteresis margin.

Control file VLLM_ADAPTIVE_K_CONTROL: a mode line "force N" | "adapt" (one state per request, Matt's rule) |
"phase", plus an optional "cost 2:ms,3:ms,..." line replacing the cost table. Missing file ->
VLLM_ADAPTIVE_K_MODE (default phase). Stats JSON -> <control>.stats.
"""

import json
import os
import time
from collections import Counter

import numpy as np

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

_ALPHA = float(os.environ.get("VLLM_ADAPTIVE_K_ALPHA", "0.25"))
_GALPHA = float(os.environ.get("VLLM_ADAPTIVE_K_GLOBAL_ALPHA", "0.02"))
_MARGIN = float(os.environ.get("VLLM_ADAPTIVE_K_MARGIN", "0.03"))
_PRIOR = 0.8

# GLM 5.3 tokenizer ids (tokenizer.json added_tokens; fences = vocab entries containing ```).
ASSISTANT, THINK, END_THINK, TOOL, END_TOOL = 154828, 154841, 154842, 154843, 154844
FENCES = frozenset((41002, 53913, 73022))
PHASES = ("reasoning", "content", "code", "tool")
# D12 k=4 conditional acceptance (LEDGER 2026-09-28 05:24Z): prose, code, structured.
SEED = {"reasoning": [.77, .66, .63, .60], "content": [.77, .66, .63, .60],
        "code": [.91, .78, .78, .73], "tool": [.99, .97, .94, .97]}


def _parse_costs(spec: str) -> tuple[np.ndarray, np.ndarray]:
    points = sorted((int(t), float(ms)) for t, ms in (p.split(":") for p in spec.split(",")))
    return np.array([p[0] for p in points], float), np.array([p[1] for p in points], float)


def _update(c: np.ndarray, drafted: int, accepted: int) -> None:
    c[:accepted] += _ALPHA * (1.0 - c[:accepted])
    if accepted < drafted:
        c[accepted] -= _ALPHA * c[accepted]
    else:
        c[drafted:] += _ALPHA * (c[drafted - 1] - c[drafted:])


class _Req:
    __slots__ = ("phase", "rates", "changed")

    def __init__(self, phase: str) -> None:
        self.phase = phase
        self.rates: dict[str, np.ndarray] = {}
        self.changed = True


def _initial_phase(prompt: list[int] | None) -> str:
    """Phase at the start of generation from the prompt tail after the last <|assistant|>."""
    phase = "content"
    if not prompt:
        return phase
    tail = prompt[-64:]
    if ASSISTANT in tail:
        tail = tail[len(tail) - 1 - tail[::-1].index(ASSISTANT):]
    for t in tail:
        phase = _next_phase(phase, t)
    return phase


def _next_phase(phase: str, t: int) -> str:
    if t == THINK:
        return "reasoning"
    if t == END_THINK or t == END_TOOL:
        return "content"
    if t == TOOL:
        return "tool"
    if t in FENCES:
        return {"content": "code", "code": "content"}.get(phase, phase)
    return phase


class AdaptiveKScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        K = self.num_spec_tokens
        # A non-None lookup turns off the scheduler's "pad new decodes to 1 + num_spec_tokens".
        self.dynamic_sd_lookup = [K] * (self.scheduler_config.max_num_seqs + 1)
        self._levels = sorted({int(x) for x in os.environ.get(
            "VLLM_ADAPTIVE_K_LEVELS", ",".join(map(str, range(1, K + 1)))).split(",") if 0 < int(x) <= K})
        self._cost_x, self._cost_y = _parse_costs(
            os.environ.get("VLLM_ADAPTIVE_K_COST_MS", "2:55,3:62,4:70,5:77"))
        self._control = os.environ.get("VLLM_ADAPTIVE_K_CONTROL", "")
        self._control_mtime = -1.0
        self._default_mode = os.environ.get("VLLM_ADAPTIVE_K_MODE", "phase")
        self._mode, self._force = self._default_mode, None
        self._reqs: dict[str, _Req] = {}
        self._prior = {p: np.array((SEED[p] + [SEED[p][-1]] * K)[:K], float) for p in PHASES}
        self._k = K
        self._steps: Counter[str] = Counter()
        self._last_log = self._last_stats = 0.0
        logger.info("adaptive k (D13): levels %s, cost %s, mode %s", self._levels,
                    dict(zip(self._cost_x.astype(int).tolist(), self._cost_y.tolist())), self._default_mode)

    # ---- observation -------------------------------------------------------------------
    def _req(self, req_id: str) -> _Req | None:
        r = self._reqs.get(req_id)
        if r is None:
            request = self.requests.get(req_id)
            if request is None:
                return None
            r = self._reqs[req_id] = _Req(_initial_phase(request.prompt_token_ids))
        return r

    def _rates(self, r: _Req, phase: str) -> np.ndarray:
        c = r.rates.get(phase)
        if c is None:
            c = r.rates[phase] = (self._prior[phase].copy() if self._mode == "phase"
                                  else np.full(self.num_spec_tokens, _PRIOR))
        return c

    def make_spec_decoding_stats(self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
                                 num_invalid_spec_tokens, request_id):
        if num_draft_tokens:
            r = self._req(request_id)
            if r is not None:
                phase = r.phase if self._mode == "phase" else "all"
                _update(self._rates(r, phase), num_draft_tokens, num_accepted_tokens)
                if r.phase in self._prior:
                    g = self._prior[r.phase]
                    g[:num_accepted_tokens] += _GALPHA * (1.0 - g[:num_accepted_tokens])
                    if num_accepted_tokens < num_draft_tokens:
                        g[num_accepted_tokens] -= _GALPHA * g[num_accepted_tokens]
                self._steps[f"obs:{r.phase}"] += 1
                self._steps[f"drafted:{r.phase}"] += num_draft_tokens
                self._steps[f"accepted:{r.phase}"] += num_accepted_tokens
        return super().make_spec_decoding_stats(spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
                                                num_invalid_spec_tokens, request_id)

    def _update_request_with_output(self, request, new_token_ids, is_stale=False):
        new_token_ids, stopped = super()._update_request_with_output(request, new_token_ids, is_stale)
        r = self._reqs.get(request.request_id)
        if r is not None:
            phase = r.phase
            for t in new_token_ids:
                phase = _next_phase(phase, t)
            if phase != r.phase:
                r.phase, r.changed = phase, True
        return new_token_ids, stopped

    # ---- choice ------------------------------------------------------------------------
    def _step_cost(self, tokens: int) -> float:
        x, y = self._cost_x, self._cost_y
        if tokens <= x[-1]:
            return float(np.interp(tokens, x, y))
        return float(y[-1] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (tokens - x[-1]))

    def _read_control(self) -> None:
        if not self._control:
            return
        try:
            mtime = os.stat(self._control).st_mtime
        except FileNotFoundError:
            mtime = 0.0
        if mtime == self._control_mtime:
            return
        self._control_mtime = mtime
        lines = open(self._control).read().splitlines() if mtime else []
        costs = [ln.split(None, 1)[1] for ln in lines if ln.startswith("cost ")]
        if costs:
            self._cost_x, self._cost_y = _parse_costs(costs[-1].strip())
            logger.info("adaptive k (D13): cost -> %s", costs[-1].strip())
        words = next((ln.split() for ln in lines if ln.split() and not ln.startswith("cost ")), [])
        mode, force = self._default_mode, None
        if len(words) == 2 and words[0] == "force":
            mode, force = "force", int(words[1])
        elif len(words) == 1 and words[0] in ("adapt", "phase"):
            mode = words[0]
        if mode != self._mode:
            for r in self._reqs.values():
                r.rates.clear()
        self._mode, self._force = mode, force
        logger.info("adaptive k (D13): control -> %s %s", mode, force or "")

    def _choose(self, req_ids: list[str]) -> int:
        if self._force is not None:
            return min(max(self._force, 1), self.num_spec_tokens)
        rs = [r for r in (self._req(i) for i in req_ids) if r is not None]
        if not rs:
            return self._k
        switched = any(r.changed for r in rs)
        for r in rs:
            r.changed = False
        stack = np.stack([self._rates(r, r.phase if self._mode == "phase" else "all") for r in rs])
        survival = np.cumprod(stack, axis=1).sum(axis=0)
        n = len(rs)
        score = {k: (n + survival[:k].sum()) / self._step_cost(n * (k + 1)) for k in self._levels}
        best = max(score, key=score.get)
        if switched and self._mode == "phase":
            return best
        current = self._k if self._k in score else best
        return best if score[best] > score[current] * (1 + _MARGIN) else current

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        for req_id in [r for r in self._reqs if r not in self.requests]:
            del self._reqs[req_id]
        if self.num_spec_tokens:
            self._read_control()
            k = self._choose(list(scheduler_output.num_scheduled_tokens))
            phases = {self._reqs[i].phase for i in scheduler_output.num_scheduled_tokens if i in self._reqs}
            for p in phases or {"-"}:
                self._steps[f"{self._mode}:{p}:k{k}"] += 1
            now = time.monotonic()
            if k != self._k and now - self._last_log > 5:
                self._last_log = now
                logger.info("adaptive k (D13): %d -> %d (%s)", self._k, k, ",".join(sorted(phases)))
            self._k = k
            scheduler_output.num_spec_tokens_to_schedule = k
            if self._control and now - self._last_stats > 10:
                self._last_stats = now
                self._write_stats()
        super()._update_after_schedule(scheduler_output)

    def _write_stats(self) -> None:
        try:
            tmp = self._control + ".stats.tmp"
            with open(tmp, "w") as f:
                json.dump({"t": time.time(), "mode": self._mode, "force": self._force, "steps": dict(self._steps),
                           "prior": {p: [round(x, 4) for x in v] for p, v in self._prior.items()}}, f)
            os.replace(tmp, self._control + ".stats")
        except OSError:
            pass
