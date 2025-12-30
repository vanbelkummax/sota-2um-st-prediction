#!/usr/bin/env python3
"""
SOTA 2μm Ablation Runner
=========================

Systematic ablation experiments following the implementation plan:

Phase 1: Encoder comparison (Week 1-2)
Phase 2: Loss function comparison (Week 2-3)
Phase 3: Decoder architecture (Week 3-4)
Phase 4: Multi-resolution fusion (Week 4-5)
Phase 5: Integration (Week 5-6)

Usage:
    python run_ablation.py --phase 1  # Encoder ablation
    python run_ablation.py --phase 2  # Loss ablation
    python run_ablation.py --phase 4  # Multi-resolution ablation
    python run_ablation.py --all      # Run all phases

Author: Max Van Belkum + Claude Opus 4.5
Date: December 2024
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import time


# Ablation configurations
PHASE_1_ENCODERS = [
    {"encoder": "prov-gigapath", "encoder_dim": 1536},
    {"encoder": "virchow2", "encoder_dim": 1280},
    {"encoder": "uni2-h", "encoder_dim": 1536},
    {"encoder": "h-optimus-1", "encoder_dim": 1024},
    {"encoder": "conchv1.5", "encoder_dim": 768},
]

PHASE_2_LOSSES = [
    {"loss": "mse"},
    {"loss": "focal", "focal_gamma": 2.0},
    {"loss": "zinb"},
    {"loss": "focal_zinb", "focal_gamma": 2.0},
    {"loss": "multitask"},
]

PHASE_3_DECODERS = [
    {"decoder": "hist2st"},
    # Add more decoders as implemented
]

PHASE_4_MULTISCALE = [
    {"scales": ["2um"], "fusion": "cross_attention"},
    {"scales": ["2um", "8um"], "fusion": "cross_attention"},
    {"scales": ["2um", "8um"], "fusion": "gated"},
    {"scales": ["2um", "8um", "32um"], "fusion": "cross_attention"},
    {"scales": ["2um", "8um", "32um"], "fusion": "hierarchical"},
]


def run_experiment(config: Dict[str, Any], base_args: List[str], output_dir: Path) -> Dict[str, Any]:
    """Run a single experiment."""
    # Build command
    cmd = [sys.executable, str(Path(__file__).parent / "train_multiscale.py")]
    cmd.extend(base_args)

    # Add config-specific arguments
    for key, value in config.items():
        if isinstance(value, list):
            cmd.extend([f"--{key}"] + [str(v) for v in value])
        else:
            cmd.extend([f"--{key}", str(value)])

    # Create experiment name
    exp_name = "_".join(f"{k}_{v}" for k, v in config.items() if k != "encoder_dim")
    exp_name = exp_name.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
    cmd.extend(["--exp_name", exp_name])

    print(f"\n{'='*60}")
    print(f"Running experiment: {exp_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Run training
    result = subprocess.run(cmd, capture_output=True, text=True)

    elapsed_time = time.time() - start_time

    # Parse results
    results_file = output_dir / exp_name / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            exp_results = json.load(f)
    else:
        exp_results = {"error": result.stderr if result.returncode != 0 else "No results file"}

    return {
        "config": config,
        "exp_name": exp_name,
        "return_code": result.returncode,
        "elapsed_minutes": elapsed_time / 60,
        "results": exp_results
    }


def run_phase(phase: int, base_args: List[str], output_dir: Path) -> List[Dict[str, Any]]:
    """Run all experiments for a phase."""
    if phase == 1:
        configs = PHASE_1_ENCODERS
        phase_name = "encoder_ablation"
    elif phase == 2:
        configs = PHASE_2_LOSSES
        phase_name = "loss_ablation"
    elif phase == 3:
        configs = PHASE_3_DECODERS
        phase_name = "decoder_ablation"
    elif phase == 4:
        configs = PHASE_4_MULTISCALE
        phase_name = "multiscale_ablation"
    else:
        raise ValueError(f"Unknown phase: {phase}")

    print(f"\n{'#'*60}")
    print(f"# Phase {phase}: {phase_name}")
    print(f"# {len(configs)} experiments to run")
    print(f"{'#'*60}\n")

    results = []
    for i, config in enumerate(configs):
        print(f"\nExperiment {i+1}/{len(configs)}")
        result = run_experiment(config, base_args, output_dir)
        results.append(result)

        # Save intermediate results
        with open(output_dir / f"phase_{phase}_results.json", 'w') as f:
            json.dump(results, f, indent=2)

    return results


def analyze_results(results: List[Dict[str, Any]], phase: int) -> Dict[str, Any]:
    """Analyze results and provide recommendations."""
    print(f"\n{'='*60}")
    print(f"Phase {phase} Results Summary")
    print(f"{'='*60}\n")

    # Extract best SSIM for each experiment
    summary = []
    for r in results:
        if "results" in r and isinstance(r["results"], dict):
            best_ssim = r["results"].get("best_ssim", 0)
        else:
            best_ssim = 0

        summary.append({
            "exp_name": r["exp_name"],
            "best_ssim": best_ssim,
            "elapsed_minutes": r.get("elapsed_minutes", 0),
            "return_code": r.get("return_code", -1)
        })

    # Sort by SSIM
    summary.sort(key=lambda x: x["best_ssim"], reverse=True)

    # Print table
    print(f"{'Experiment':<50} {'SSIM':<10} {'Time (min)':<10} {'Status'}")
    print("-" * 80)
    for s in summary:
        status = "OK" if s["return_code"] == 0 else "FAILED"
        print(f"{s['exp_name']:<50} {s['best_ssim']:.4f}     {s['elapsed_minutes']:.1f}        {status}")

    # Recommendations
    if summary and summary[0]["best_ssim"] > 0:
        best = summary[0]
        print(f"\nBest configuration: {best['exp_name']}")
        print(f"Best SSIM: {best['best_ssim']:.4f}")

        # Decision logic
        if len(summary) >= 2:
            second = summary[1]
            improvement = best['best_ssim'] - second['best_ssim']

            if improvement < 0.02:
                print(f"\nNote: Top 2 configs differ by only {improvement:.4f} SSIM")
                print("Consider using simpler/faster option if available")

    return {
        "phase": phase,
        "summary": summary,
        "best_config": summary[0] if summary else None
    }


def main():
    parser = argparse.ArgumentParser(description="SOTA 2μm Ablation Runner")

    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5],
                        help="Phase to run (1=encoder, 2=loss, 3=decoder, 4=multiscale)")
    parser.add_argument("--all", action="store_true", help="Run all phases")

    # Base training arguments
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_patients", nargs="+", default=["P1", "P2"])
    parser.add_argument("--test_patient", type=str, default="P5")
    parser.add_argument("--output_dir", type=str,
                        default="/home/user/sota-2um-st-prediction/results/ablation")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base arguments for all experiments
    base_args = [
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--train_patients", *args.train_patients,
        "--test_patient", args.test_patient,
        "--output_dir", str(output_dir),
        "--amp",
    ]

    all_results = {}
    phases_to_run = [1, 2, 3, 4] if args.all else [args.phase] if args.phase else []

    if not phases_to_run:
        print("Specify --phase N or --all")
        return

    for phase in phases_to_run:
        results = run_phase(phase, base_args, output_dir)
        analysis = analyze_results(results, phase)
        all_results[f"phase_{phase}"] = analysis

        # Save cumulative results
        with open(output_dir / "all_ablation_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Ablation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
