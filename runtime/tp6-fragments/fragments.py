"""Exact TP4-shard placement on six ranks; no weight de/requantization.

Each logical expert keeps all four original 512-channel fragments. A rank
owns at most one fragment of an expert. Missing routes contribute zero.
All ranks receive the same tokens/router choices and sum partial outputs.
"""
import re
from functools import lru_cache

EXPERTS = 256
SOURCE_TP = 4
WORLD = 6
FRAGMENT_WIDTH = 512
PATTERN = re.compile(r'^(.*layers\.(\d+)\.mlp\.experts\.)(\d+)\.(gate_proj|up_proj|down_proj)\.rank(\d+)\.(trellis|suh|svh|mcg)$')


def owner(expert, part, layer=0):
    if not 0 <= expert < EXPERTS or not 0 <= part < SOURCE_TP:
        raise ValueError((expert, part))
    # Rotate the four-of-six placement by layer, not the source payload.
    return (4 * expert + part + layer) % WORLD


@lru_cache(None)
def placement(rank, layer):
    if not 0 <= rank < WORLD:
        raise ValueError(rank)
    return tuple((e, p) for e in range(EXPERTS) for p in range(SOURCE_TP)
                 if owner(e, p, layer) == rank)


def normalize(name, rank):
    match = PATTERN.fullmatch(name)
    if match is None:
        return name
    prefix, layer, expert, projection, part, field = match.groups()
    layer, expert, part = int(layer), int(expert), int(part)
    if owner(expert, part, layer) != rank:
        return None
    local = placement(rank, layer).index((expert, part))
    return f'{prefix}{local}.{projection}.{field}'
