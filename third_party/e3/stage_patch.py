#!/usr/bin/env python3
"""Produce a COPY-only candidate overlay; never edits the running server."""
import argparse
from pathlib import Path


def patch(source, stream_scoped=False):
    anchor = '''    def _apply_mixed_rank_sliced(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        runtime = self._mixed_rank_sliced_runtime(layer, x, topk_ids)
'''
    assert source.count(anchor) == 1
    result = source.replace(anchor, anchor + '''        # Stock runtime planning above is required by decode route-pack warmup.
        # Experimental prefill only; uniform decode remains on qualified B12X.
        if os.environ.get("GLM6_E3_PREFILL", "0") == "1" and int(x.shape[0]) > 32:
            from e3.runtime import apply as glm6_e3_apply
            return glm6_e3_apply(layer, x, topk_weights, topk_ids)
''')
    if stream_scoped:
        result = result.replace('            from e3.runtime import apply as glm6_e3_apply\n            return glm6_e3_apply(layer, x, topk_weights, topk_ids)',
            '            from e3.serving import can_apply, apply as glm6_e3_apply\n            if can_apply(layer, x, topk_ids):\n                return glm6_e3_apply(layer, x, topk_weights, topk_ids)')
    compile(result, 'exl3-e3-candidate.py', 'exec')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream-scoped', action='store_true')
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit('Refusing to overwrite candidate source')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(patch(Path(args.source).read_text(), stream_scoped=args.stream_scoped))
