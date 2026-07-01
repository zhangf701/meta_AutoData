"""ID Manager: Auto-increment question IDs per level.

Tracks the next available ID for each level (L1/L2/L3).
"""


class IDManager:
    """Generate sequential question IDs, starting from specified offsets."""

    def __init__(self, l1_start=101, l2_start=101, l3_start=101):
        """Start at 101 to avoid collision with original eval_set_70 (1-24 for L1/L2, 1-12 for L3)."""
        self.counters = {"L1": l1_start, "L2": l2_start, "L3": l3_start}

    def next(self, level):
        """Return next ID for given level, e.g., 'L1-101'."""
        if level not in self.counters:
            level = "L2"
        cid = f"{level}-{self.counters[level]:03d}"
        self.counters[level] += 1
        return cid

    def current_counts(self):
        """Return current count per level."""
        return dict(self.counters)
