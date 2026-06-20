# To save time, this part was made with Claude AI based on the both training files
# For better evaluation this file was created to train both models on the same maps

"""
CombinedMetrics.py

Pretty-prints a Q-learning vs VFA comparison summary along the three axes
the project brief asks you to compare: convergence speed, sample
efficiency, and policy quality. Works directly off the dicts returned by
train_combined() / evaluate_combined(), or off the CSV files they save to
disk (so you can print a summary later without re-running training).

Usage straight after training/eval:

    from training.CombinedTrain import train_combined
    from evaluation.CombinedEvaluation import evaluate_combined
    from CombinedMetrics import print_comparison

    train_metrics = train_combined(env, Qagent, VFAagent,
                                    num_episodes=num_episodes,
                                    max_training_steps=max_training_steps)
    eval_metrics = evaluate_combined(env, Qagent, VFAagent)

    print_comparison(train_metrics, eval_metrics)

Usage later, from saved CSVs only (no agents/env needed):

    from CombinedMetrics import load_metrics_from_csv, print_comparison

    train_metrics = load_metrics_from_csv("output/comparison/training_metrics.csv")
    eval_metrics = load_metrics_from_csv("output/comparison/evaluation_metrics.csv")
    print_comparison(train_metrics, eval_metrics)
"""

import csv

import numpy as np


def load_metrics_from_csv(path, q_prefix="q_", vfa_prefix="vfa_"):
    """Re-load a metrics CSV (written by train_combined / evaluate_combined)
    back into the {"q": {...}, "vfa": {...}} dict format."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def col(prefix, name):
        key = f"{prefix}{name}"
        return np.array([row[key] for row in rows], dtype=float)

    return {
        "q": {
            "returns": col(q_prefix, "return"),
            "steps": col(q_prefix, "steps"),
            "success": col(q_prefix, "success").astype(bool),
        },
        "vfa": {
            "returns": col(vfa_prefix, "return"),
            "steps": col(vfa_prefix, "steps"),
            "success": col(vfa_prefix, "success").astype(bool),
        },
    }


def _summarize(metrics, window):
    out = {}
    for algo in ("q", "vfa"):
        returns = metrics[algo]["returns"]
        steps = metrics[algo]["steps"]
        success = metrics[algo]["success"].astype(bool)
        n = len(returns)
        w = min(window, n) if n else 0

        first_success = int(np.argmax(success)) if success.any() else None
        succ_steps = steps[success]

        out[algo] = {
            "n": n,
            "success_rate_overall": success.mean() if n else float("nan"),
            "success_rate_final_window": success[-w:].mean() if w else float("nan"),
            "avg_return_overall": returns.mean() if n else float("nan"),
            "avg_return_final_window": returns[-w:].mean() if w else float("nan"),
            "first_success_episode": first_success,
            "avg_steps_when_successful": succ_steps.mean() if len(succ_steps) else float("nan"),
        }
    return out


def print_comparison(train_metrics=None, eval_metrics=None, window=500):
    """
    Print a side-by-side Q-learning vs VFA comparison summary.

    Parameters
    ----------
    train_metrics : dict from train_combined() (or load_metrics_from_csv()
                    on the training CSV), or None to skip training stats
    eval_metrics  : dict from evaluate_combined() (or load_metrics_from_csv()
                    on the evaluation CSV), or None to skip eval stats
    window        : trailing-episode window used for "final" training
                    performance, kept separate from the all-episode average
                    (which is dragged down by early, mostly-random episodes)
    """
    line = "=" * 72
    print(line)
    print("Q-LEARNING vs VFA -- COMPARISON SUMMARY")
    print(line)

    if train_metrics is not None:
        s = _summarize(train_metrics, window)
        n = s["q"]["n"]
        w = min(window, n) if n else 0
        print(f"\nTRAINING ({n} episodes; \"final\" = last {w} episodes)"
              "  [convergence speed / sample efficiency]")
        print(f"{'':30s}{'Q-learning':>18s}{'VFA':>18s}")
        print(f"{'first success at episode':30s}"
              f"{str(s['q']['first_success_episode']):>18s}"
              f"{str(s['vfa']['first_success_episode']):>18s}")
        print(f"{'success rate (overall)':30s}"
              f"{100*s['q']['success_rate_overall']:>17.1f}%"
              f"{100*s['vfa']['success_rate_overall']:>17.1f}%")
        print(f"{'success rate (final window)':30s}"
              f"{100*s['q']['success_rate_final_window']:>17.1f}%"
              f"{100*s['vfa']['success_rate_final_window']:>17.1f}%")
        print(f"{'avg return (overall)':30s}"
              f"{s['q']['avg_return_overall']:>18.1f}"
              f"{s['vfa']['avg_return_overall']:>18.1f}")
        print(f"{'avg return (final window)':30s}"
              f"{s['q']['avg_return_final_window']:>18.1f}"
              f"{s['vfa']['avg_return_final_window']:>18.1f}")
        print(f"{'avg steps when successful':30s}"
              f"{s['q']['avg_steps_when_successful']:>18.1f}"
              f"{s['vfa']['avg_steps_when_successful']:>18.1f}")

    if eval_metrics is not None:
        n_eval = len(eval_metrics["q"]["returns"])
        s = _summarize(eval_metrics, window=n_eval)
        print(f"\nEVALUATION ({n_eval} greedy episodes, shared maps)"
              "  [policy quality]")
        print(f"{'':30s}{'Q-learning':>18s}{'VFA':>18s}")
        print(f"{'success rate':30s}"
              f"{100*s['q']['success_rate_overall']:>17.1f}%"
              f"{100*s['vfa']['success_rate_overall']:>17.1f}%")
        print(f"{'avg return':30s}"
              f"{s['q']['avg_return_overall']:>18.1f}"
              f"{s['vfa']['avg_return_overall']:>18.1f}")
        print(f"{'avg steps when successful':30s}"
              f"{s['q']['avg_steps_when_successful']:>18.1f}"
              f"{s['vfa']['avg_steps_when_successful']:>18.1f}")

    print(line)