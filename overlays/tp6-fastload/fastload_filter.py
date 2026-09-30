"""Skip redundant raw expert iteration only when direct-tier loading owns it.

The existing fastload.attach/restore path independently indexes, validates and
loads every required packed tensor. This filter avoids creating individual
CPU tensor views and walking expert mappings that cannot populate the serving
weights anyway. Dense, shared-expert, attention and unknown keys still take
the original loader. No change to model execution or on-disk weights.
"""
import os
from fragments import PATTERN


def wrap_skip_weight(original):
    """Preserve the original EP predicate and capture launch-time opt-outs."""
    enabled = (os.environ.get('GLM6_FAST_LOAD', '0') == '1'
               and os.environ.get('GLM6_FAST_FILTER', '0') == '1')

    def should_skip_weight(name, local_expert_ids):
        if enabled and PATTERN.fullmatch(name) is not None:
            return True
        return original(name, local_expert_ids)

    return should_skip_weight
