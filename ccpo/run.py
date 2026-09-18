#!/usr/bin/env python3
"""Launch a main-method arm: CCPO-ATTNCRED, context advantage, or future outlook.

Four main experiments per method (experiments/experiments.md):

    attncred                     x {alfworld, webshop} x {1.5b, 7b}
    attncred-context-adv-only    x {alfworld, webshop} x {1.5b, 7b}

Usage
    python3 official-repo/ccpo/run.py --method attncred \
        --benchmark alfworld --backbone 1.5b --gpus 0,1,2,3

    python3 official-repo/ccpo/run.py --method attncred-context-adv-only \
        --benchmark webshop --backbone 7b --gpus 0,1,2,3,4,5,6,7 --dry-run

    python3 official-repo/ccpo/run.py --list

Unrecognised arguments go to scripts/exp_run.py verbatim, and a later --set wins, so
`--set val_batch_size=64` adapts to a host without editing the arm.
"""
import argparse
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import arms                                                   # noqa: E402


def describe():
    print("methods:")
    for m, delta in arms.METHODS.items():
        d = ", ".join(f"{k}={v}" for k, v in delta.items()) or "(the published arm)"
        print(f"  {m:36s} {d:44s} {'+'.join(arms.benchmarks_for(m))}")
    print("benchmarks:")
    for b, delta in arms.BENCHMARK.items():
        print(f"  {b:28s} env={delta['env_name']}, max_steps={delta['max_steps']}, "
              f"target={delta['ccpo_target']}")
    print("backbones:")
    for b, d in arms.BACKBONE.items():
        print(f"  {b:28s} {d['model']}  (>= {d['min_gpus']} GPUs)")
    total = sum(len(arms.BACKBONE) * len(arms.benchmarks_for(m)) for m in arms.METHODS)
    print(f"\nall {total} main runs:")
    for m in arms.METHODS:
        for b in arms.benchmarks_for(m):
            print(f"  {arms.variant_name(m, b)}")
            for k in arms.BACKBONE:
                name, _ = arms.build(m, b, k)
                print(f"      {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 epilog="extra arguments are forwarded to exp_run.py")
    ap.add_argument("--method", choices=sorted(arms.METHODS), default="attncred")
    ap.add_argument("--backbone", choices=sorted(arms.BACKBONE), default="1.5b")
    ap.add_argument("--list", action="store_true", help="print the arm matrix and exit")
    arms.add_common_args(ap)
    # --list must work without the required arguments
    if "--list" in sys.argv:
        describe()
        return 0
    a, passthrough = ap.parse_known_args()

    allowed = arms.benchmarks_for(a.method)
    if a.benchmark not in allowed:
        ap.error(f"{a.method} is defined for {', '.join(allowed)} only "
                 f"(see --list); got --benchmark {a.benchmark}")
    name, cfg = arms.build(a.method, a.benchmark, a.backbone)
    arms.check_gpus(a.gpus, a.backbone)
    return arms.launch(a.name or name, cfg, a.benchmark, a.gpus, passthrough,
                       allow_sdpa=a.allow_sdpa,
                       label=f"{arms.variant_name(a.method, a.benchmark)}  "
                             f"({a.method}, {a.benchmark}, {a.backbone})")


if __name__ == "__main__":
    sys.exit(main())
