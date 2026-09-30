"""Staged at PYTHONPATH root so exec/spawn workers inherit the same opt-in hook."""
import os
import importlib.machinery
import importlib.util
from pathlib import Path
import sys


def _chain_original():
    """Run an existing sitecustomize instead of silently shadowing it."""
    here = Path(__file__).resolve().parent
    paths = [p for p in sys.path if Path(p or '.').resolve() != here]
    spec = importlib.machinery.PathFinder.find_spec('sitecustomize', paths)
    if spec is not None and spec.loader is not None:
        previous = sys.modules['sitecustomize']
        module = importlib.util.module_from_spec(spec)
        sys.modules['sitecustomize'] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules['sitecustomize'] = previous
            raise


_chain_original()

if os.environ.get('TP4_E3_PREFILL', '0') == '1':
    try:
        from e3.hook import register
        register()
    except Exception as exc:
        # Python normally logs sitecustomize exceptions and continues unpatched.
        raise SystemExit(f'TP4 E3 bootstrap failed: {exc}') from exc
