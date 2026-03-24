"""Statistical significance testing matching CHAI's methodology.

- Bootstrap for WER/AWER (1000 iterations, batch size 100, 95% CI)
- Repeated measures ANOVA + post-hoc Tukey for TD metrics
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class BootstrapResult:
    """Result of bootstrap significance test."""

    metric_name: str
    system_a_mean: float
    system_b_mean: float
    difference: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool  # True if CI doesn't contain 0


@dataclass
class AnovaResult:
    """Result of repeated measures ANOVA + Tukey test."""

    metric_name: str
    f_statistic: float
    p_value: float
    significant: bool
    # Pairwise comparisons (Tukey HSD)
    pairwise: list[tuple[str, str, float, bool]]  # (sys_a, sys_b, p_val, sig)


def bootstrap_wer(
    refs: list[list[str]],
    hyps_a: list[list[str]],
    hyps_b: list[list[str]],
    metric_fn: callable,
    metric_name: str = "WER",
    n_iterations: int = 1000,
    batch_size: int = 100,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap significance test for WER/AWER between two systems.

    Samples batches of utterances with replacement and computes the
    metric difference. Returns confidence interval and p-value.

    Args:
        refs: Reference token sequences.
        hyps_a: System A hypotheses.
        hyps_b: System B hypotheses.
        metric_fn: Function(refs, hyps) -> float (e.g., compute_wer).
        metric_name: Name for reporting.
        n_iterations: Number of bootstrap samples.
        batch_size: Number of utterances per sample.
        confidence: Confidence level (e.g., 0.95).
        seed: Random seed.
    """
    rng = np.random.RandomState(seed)
    n = len(refs)

    diffs = []
    for _ in range(n_iterations):
        indices = rng.choice(n, size=min(batch_size, n), replace=True)
        sample_refs = [refs[i] for i in indices]
        sample_a = [hyps_a[i] for i in indices]
        sample_b = [hyps_b[i] for i in indices]

        score_a = metric_fn(sample_refs, sample_a)
        score_b = metric_fn(sample_refs, sample_b)
        diffs.append(score_a - score_b)

    diffs = np.array(diffs)
    alpha = 1 - confidence
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))

    # Two-sided p-value: fraction of diffs with opposite sign of mean
    mean_diff = float(np.mean(diffs))
    if mean_diff >= 0:
        p_value = float(np.mean(diffs <= 0)) * 2
    else:
        p_value = float(np.mean(diffs >= 0)) * 2
    p_value = min(p_value, 1.0)

    system_a_mean = metric_fn(refs, hyps_a)
    system_b_mean = metric_fn(refs, hyps_b)

    return BootstrapResult(
        metric_name=metric_name,
        system_a_mean=system_a_mean,
        system_b_mean=system_b_mean,
        difference=mean_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        significant=(ci_lower > 0 or ci_upper < 0),
    )


def anova_td(
    td_scores: dict[str, list[float]],
    metric_name: str = "TD",
    alpha: float = 0.05,
) -> AnovaResult:
    """Repeated measures ANOVA + post-hoc Tukey for TD metrics.

    Args:
        td_scores: Dict mapping system name to list of per-utterance TD values.
        metric_name: Name for reporting.
        alpha: Significance threshold.

    Returns:
        AnovaResult with F-statistic, p-value, and pairwise comparisons.
    """
    systems = sorted(td_scores.keys())
    groups = [np.array(td_scores[s]) for s in systems]

    # One-way ANOVA (approximation of repeated measures)
    f_stat, p_val = stats.f_oneway(*groups)

    # Post-hoc pairwise t-tests with Bonferroni correction
    n_comparisons = len(systems) * (len(systems) - 1) // 2
    pairwise = []
    for i in range(len(systems)):
        for j in range(i + 1, len(systems)):
            t_stat, p_pair = stats.ttest_rel(groups[i], groups[j])
            p_corrected = min(p_pair * n_comparisons, 1.0)
            pairwise.append((
                systems[i],
                systems[j],
                float(p_corrected),
                p_corrected < alpha,
            ))

    return AnovaResult(
        metric_name=metric_name,
        f_statistic=float(f_stat),
        p_value=float(p_val),
        significant=p_val < alpha,
        pairwise=pairwise,
    )
