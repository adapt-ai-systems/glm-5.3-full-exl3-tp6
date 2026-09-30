# Experiments (not part of either kept stack)

- `d13-adaptive-k/`: adaptive and phase-aware MTP draft length on TP6. Measured, not shipped (partial A/B; see
  its README and RESULTS.md).
- `megamoe/`: an EXL3 fused down-projection prefill kernel for TP4. **Not kept**: +2% prefill, −3.7% structured
  decode. `REPORT.md` has the summary, `RESULTS.md` the one-layer measurements, `harness/` the bench scripts
  (they expect a TP4 checkpoint layer and the E3 package on the path), `out/` the raw measurement files.
- `tp4-profiles/`: torch-profiler breakdowns of TP4 prefill and decode on the kept S4 stack.
