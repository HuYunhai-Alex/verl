# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0

from collections import Counter
from typing import Any, Dict, Iterable, List, Set
import re
import string

# ---------- Simple EM helpers (kept for potential reuse) ----------
def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation/articles, and trim whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def em_check(prediction: str, golden_answers: Iterable[str]) -> int:
    """Exact match after normalization."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    p = normalize_answer(prediction)
    return int(any(normalize_answer(g) == p for g in golden_answers))

def subem_check(prediction: str, golden_answers: Iterable[str]) -> int:
    """Substring-EM after normalization."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    p = normalize_answer(prediction)
    return int(any(normalize_answer(g) in p for g in golden_answers))

# ---------- Extraction utility (kept for compatibility) ----------
def extract_solution(solution_str: str) -> str | None:
    """Return the last <answer>...</answer> block if present, else raw tail."""
    m = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    if not m:
        return solution_str
    return m[-1].group(1).strip()

# ---------- ID utilities for message-style solutions ----------
def _iter_message_ids(items: List[Dict[str, Any]]) -> List[str]:
    """Yield string IDs for dicts with type in {message, messgae} (typo-tolerant)."""
    ids: List[str] = []
    for d in items:
        if not isinstance(d, dict):
            continue
        t = str(d.get("type", "")).lower()
        if t in {"message", "messgae"} and d.get("id") is not None:
            ids.append(str(d["id"]))
    return ids

def gt_coverage(solution_items: List[Dict[str, Any]], gt_ids: Iterable[str]) -> Dict[str, float]:
    """Coverage against GT (set-based)."""
    gt: Set[str] = {str(x) for x in gt_ids if x is not None}
    ids = set(_iter_message_ids(solution_items))
    covered = gt & ids
    return {
        "covered_count": float(len(covered)),
        "gt_size": float(len(gt)),
        "coverage_rate": 1.0 if not gt else len(covered) / len(gt),
    }

def match_ratio(solution_items: List[Dict[str, Any]], gt_ids: Iterable[str]) -> float:
    """Multiset intersection / |GT|, considering duplicates in solution."""
    gt_counter = Counter(str(x) for x in gt_ids if x is not None)
    sol_counter = Counter(_iter_message_ids(solution_items))
    if not gt_counter:
        return 1.0
    hits = sum(min(sol_counter[k], gt_counter[k]) for k in gt_counter)
    return hits / sum(gt_counter.values())

def penalized_solution_score(
    solution_items: List[Dict[str, Any]],
    gt_ids: Iterable[str],
    *,
    alpha: float = 0.5,  # weight for irrelevant proportion
    beta: float = 0.5,   # weight for duplicate-correct proportion
    enable_penalize: bool = True
) -> float:
    """
    Score = coverage_rate - alpha * irrel_rate - beta * dup_correct_rate, clamped to [0, 1].

    - coverage_rate: unique coverage of GT (|covered(GT∩sol)| / |GT|)
    - irrel_rate: fraction of solution occurrences not in GT
    - dup_correct_rate: fraction of extra repeats for correct items
    """
    gt_set: Set[str] = {str(x) for x in gt_ids if x is not None}
    ids = _iter_message_ids(solution_items)
    N = len(ids)

    # Coverage (unique)
    coverage_rate = 1.0 if not gt_set else len(gt_set & set(ids)) / len(gt_set)
    if not enable_penalize:
        return max(0.0, min(1.0, coverage_rate))

    cnt = Counter(ids)
    n_irrel = sum(c for k, c in cnt.items() if k not in gt_set)
    irrel_rate = 0.0 if N == 0 else n_irrel / N

    dup_correct = sum(max(c - 1, 0) for k, c in cnt.items() if k in gt_set)
    dup_correct_rate = 0.0 if N == 0 else dup_correct / N

    score = coverage_rate - alpha * irrel_rate - beta * dup_correct_rate
    return max(0.0, min(1.0, score))

# ---------- Main API (kept signature-compatible) ----------
def compute_score(
    data_source: Any,
    solution_str: Dict[str, Any] | None,
    ground_truth: Dict[str, Any],
    extra_info: Dict[str, Any] | None = None
) -> float:
    """
    Scoring entry:
    - Expects solution_str like {"result": [ {...}, ... ], "monc_sql": "<json string>"}
    - ground_truth: {"ground_truth": [id,...], "rel_requirements": optional ...}
    - Returns float in [0,1]; returns -1 if solution missing when relational requirements exist.
    """
    # Missing or empty solution handling
    enable_penalize = ground_truth.get("rel_requirements") is not None
    
    if solution_str is None:
        return -1.0 if enable_penalize else 0.0
    if not solution_str.get("result"):
        return 0.0

    solution_items = solution_str["result"]
    gt_ids = ground_truth.get("ground_truth", [])
    score = penalized_solution_score(
        solution_items,
        gt_ids,
        alpha=0.5,
        beta=0.5,
        enable_penalize=enable_penalize,
    )

    print(f"Computed score: {score:.4f} (enable_penalize={enable_penalize}), monc_sql: {solution_str.get('monc_sql', '')}, solution_items: {len(solution_items)}, gt_ids: {len(gt_ids)}")
    return score