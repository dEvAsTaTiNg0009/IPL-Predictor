#!/usr/bin/env python3
"""
Fast Leak-Free IPL Prediction Evaluator.
Runs sequential walk-forward backtesting across recent IPL seasons without temporal leakage.
"""

import argparse
from pathlib import Path
from tabulate import tabulate
from walk_forward_backtest import WalkForwardBacktester


def main():
    parser = argparse.ArgumentParser(description="Fast Leak-Free IPL Backtest Evaluator")
    parser.add_argument("--mode", type=str, default="pre_xi", choices=["pre_xi", "post_xi"])
    parser.add_argument("--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025, 2026])
    args = parser.parse_args()

    backtester = WalkForwardBacktester(mode=args.mode)
    backtester.load_data()

    print(f"\nEvaluating seasons: {args.seasons} in {args.mode.upper()} mode...")
    season_results, match_preds = backtester.run_walk_forward(test_seasons=args.seasons)

    total_matches = sum(r["matches"] for r in season_results)
    total_correct = sum(r["correct"] for r in season_results)
    overall_acc = total_correct / total_matches if total_matches > 0 else 0.0

    print("\n" + "=" * 65)
    print("  LEAK-FREE WALK-FORWARD EVALUATION RESULTS")
    print("=" * 65)

    rows = []
    for r in season_results:
        rows.append([
            f"{r['train_start']}–{r['train_end']}",
            r["test_season"],
            r["matches"],
            r["correct"],
            f"{r['accuracy']:.1%}",
            f"{r['log_loss']:.4f}",
            f"{r['brier_score']:.4f}",
            f"{r['roc_auc']:.4f}",
        ])

    print(tabulate(
        rows,
        headers=["Train Window", "Test Season", "Matches", "Correct", "Accuracy", "Log Loss", "Brier", "ROC-AUC"],
        tablefmt="rounded_outline",
    ))

    print(f"\n  Overall Accuracy: {overall_acc:.1%} ({total_correct}/{total_matches} matches)")
    print(f"  All features strictly pre-match | Synthetic Data: False | Pre-XI Mode\n")


if __name__ == "__main__":
    main()
