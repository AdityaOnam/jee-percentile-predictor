"""Command line interface: python -m jee_percentile <command>."""

import argparse
import sys

from jee_percentile.config import ARTIFACT_PATH


def _predict(args):
    from jee_percentile.portable import load_artifact, predict_curve, predict_percentile

    if not ARTIFACT_PATH.exists():
        sys.exit("No model artifact yet - run `python -m jee_percentile train` first.")
    art = load_artifact(ARTIFACT_PATH)
    low, pct, high = predict_percentile(art, args.marks, args.physics_hard, args.physics_easy)
    percentiles, curve = predict_curve(art, args.physics_hard, args.physics_easy)
    if args.marks >= curve[-1]:
        print(f"\n  {args.marks:g}/300 marks  ->  99+ percentile (above the modelled 90-99 range)\n")
    elif args.marks <= curve[0]:
        print(f"\n  {args.marks:g}/300 marks  ->  below 90 percentile (under the modelled range)\n")
    else:
        print(f"\n  {args.marks:g}/300 marks  ->  {pct:.2f} percentile")
        print(f"  80% band: {low:.2f} - {high:.2f}\n")
    print("  Marks needed for each percentile on this paper:")
    for p, m in zip(percentiles, curve):
        bar = "#" * max(0, int(m / 5))
        print(f"    {p:>5.1f}  {m:6.1f}  {bar}")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jee-pct", description="JEE Main percentile predictor")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="estimate percentile for a score")
    p.add_argument("--marks", type=float, required=True, help="raw score out of 300")
    p.add_argument("--physics-hard", type=float, required=True, help="%% of Physics questions labelled hard")
    p.add_argument("--physics-easy", type=float, required=True, help="%% of Physics questions labelled easy")
    sub.add_parser("train", help="fit on all shifts and export models/percentile_model_artifact.json")
    sub.add_parser("evaluate", help="LOSO-CV, baselines, ablations -> reports/")
    sub.add_parser("figures", help="render every chart into reports/figures/")
    sub.add_parser("all", help="train + evaluate + figures")

    args = parser.parse_args(argv)
    if args.command == "predict":
        _predict(args)
        return
    if args.command in ("train", "all"):
        from jee_percentile.train import export
        export()
        print(f"model -> {ARTIFACT_PATH}")
    if args.command in ("evaluate", "all"):
        from jee_percentile.evaluate import run
        for row in run()["leaderboard"]:
            print(f"{row['name']:30s} percentile MAE {row['pct_mae']:.3f}")
    if args.command in ("figures", "all"):
        from jee_percentile.plots import render_all
        for path in render_all():
            print(f"figure -> {path}")


if __name__ == "__main__":
    main()
