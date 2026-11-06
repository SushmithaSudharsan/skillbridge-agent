"""Metrics for judging an LLM system. Plain Python so every formula is visible.

Each function matches a section of the guide's Metrics reference tab.
"""

from itertools import combinations
from math import comb, sqrt


# ---------- Finding the right items in a set ----------

def precision_recall_f1(predicted: set, expected: set) -> tuple[float, float, float]:
    """precision = TP / (TP + FP); recall = TP / (TP + FN); F1 = harmonic mean.
    Nothing predicted and nothing expected counts as perfect (1.0)."""
    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else 1.0
    recall = tp / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def micro_prf(pairs: list[tuple[set, set]]) -> tuple[float, float, float]:
    """Pool TP/FP/FN over every example first, then compute once."""
    tp = sum(len(p & e) for p, e in pairs)
    fp = sum(len(p - e) for p, e in pairs)
    fn = sum(len(e - p) for p, e in pairs)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def jaccard(a: set, b: set) -> float:
    """Overlap of two sets: |A and B| / |A or B|. Two empty sets are identical (1.0)."""
    return len(a & b) / len(a | b) if a | b else 1.0


def consistency(runs: list[set]) -> float:
    """Mean Jaccard over every pair of runs. 1.0 = the same answer every time."""
    if len(runs) < 2:
        return 1.0
    pairs = list(combinations(runs, 2))
    return sum(jaccard(a, b) for a, b in pairs) / len(pairs)


# ---------- Do two graders agree beyond chance? ----------

def cohen_kappa(a: list, b: list, weights: str | None = None) -> float:
    """kappa = (p_o - p_e) / (1 - p_e). weights='quadratic' for ordered scores like 1-3."""
    if len(a) != len(b) or not a:
        raise ValueError("Need two equal-length, non-empty lists")
    labels = sorted(set(a) | set(b))
    index = {label: i for i, label in enumerate(labels)}
    k, n = len(labels), len(a)

    def weight(i: int, j: int) -> float:                      # disagreement weight
        if weights == "quadratic":
            return ((i - j) / (k - 1)) ** 2 if k > 1 else 0.0
        return 0.0 if i == j else 1.0

    observed = sum(weight(index[x], index[y]) for x, y in zip(a, b)) / n
    pa = [sum(1 for x in a if x == label) / n for label in labels]
    pb = [sum(1 for y in b if y == label) / n for label in labels]
    expected = sum(pa[i] * pb[j] * weight(i, j) for i in range(k) for j in range(k))
    return 1.0 if expected == 0 else 1 - observed / expected


# ---------- Does the agent succeed reliably? ----------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Chance that at least one of k runs succeeds, from n runs with c successes."""
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k: chance that ALL k runs succeed. What a user who gets one run cares about."""
    return comb(c, k) / comb(n, k) if c >= k else 0.0


# ---------- Is B better than A? ----------

def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a win rate. Wide on small samples, on purpose."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    centre = p + z * z / (2 * n)
    spread = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    denominator = 1 + z * z / n
    return (centre - spread) / denominator, (centre + spread) / denominator


# ---------- How different are two orderings? (used by the debate, M6) ----------

def kendall_tau_distance(a: list, b: list) -> float:
    """Share of item pairs the two orderings put in opposite order. 0 = same, 1 = reversed.
    Only items in both lists are compared."""
    common = [x for x in a if x in b]
    pairs = list(combinations(common, 2))
    if not pairs:
        return 0.0
    rank_b = {x: i for i, x in enumerate(b)}
    flipped = sum(1 for x, y in pairs if rank_b[x] > rank_b[y])
    return flipped / len(pairs)
