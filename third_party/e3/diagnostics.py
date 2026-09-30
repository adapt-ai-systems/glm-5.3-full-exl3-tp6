"""Opt-in CPU/GPU breadcrumbs for standalone E3 isolation, never auto-enabled."""
import json
import time


def validate_segments(routes, experts, rows):
    """CPU lists only: validates actual occupied segments, not padded suffix."""
    n = int(routes['num_segs'][0]); nr = int(routes['num_rows'][0])
    assert 0 <= nr <= rows and 0 <= n <= len(routes['seg_expert'])
    total = 0
    for e, start, count in zip(routes['seg_expert'][:n], routes['seg_row0'][:n], routes['seg_rows'][:n]):
        assert 0 <= e < experts and 0 < count <= 64
        assert 0 <= start and start + count <= nr
        total += count
    assert total == nr
    return {'actual_segments': n, 'actual_rows': nr, 'padded_segments': len(routes['seg_expert'])}


class Probe:
    """Instrument only an explicit synchronized diagnostic call.

    Host launch is logged BEFORE launch; completion AFTER event synchronization.
    Never use these timings as asynchronous-performance evidence.
    """
    def __init__(self, path, label):
        self.path = path; self.label = label
    def emit(self, event, **data):
        with open(self.path, 'a') as f:
            f.write(json.dumps(dict(utc_epoch=time.time(), label=self.label, event=event, **data))+'\n')
    def __enter__(self):
        import torch
        from . import runtime
        self.original = runtime.DeviceModule.launch
        self.route_original = runtime.route_tables
        outer = self
        def route(ids, weights, mapping, experts):
            r = outer.route_original(ids, weights, mapping, experts)
            cpu = {k: r[k].cpu().tolist() for k in ('num_rows','num_segs','seg_expert','seg_row0','seg_rows')}
            outer.emit('ROUTES_VALIDATED', **validate_segments(cpu, experts, ids.numel()))
            return r
        def launch(module, name, grid, args, stream, smem=0):
            outer.emit('HOST_LAUNCH', phase=name, grid=grid, stream=stream,
                       capture=torch.cuda.is_current_stream_capturing())
            outer.original(module, name, grid, args, stream, smem)
            event = torch.cuda.Event(); event.record(); event.synchronize()
            outer.emit('DEVICE_COMPLETED', phase=name, stream=stream)
        runtime.DeviceModule.launch = launch; runtime.route_tables = route
        return self
    def __exit__(self, *exc):
        from . import runtime
        runtime.DeviceModule.launch = self.original
        runtime.route_tables = self.route_original
