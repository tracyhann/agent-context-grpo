#!/usr/bin/env python3
"""Launch one CCPO-ATTNCRED ablation. 1.5B or 7B, both benchmarks, 150 steps.

Usage
    python3 official-repo/ablations/run.py --ablation hard-gate \
        --benchmark alfworld --gpus 0,1,2,3
    python3 official-repo/ablations/run.py --list

Each ablation changes its declared base method on the same benchmark;
`--list` prints the key. Extra arguments are forwarded to scripts/exp_run.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "ccpo"))
import ablations                                              # noqa: E402
import arms                                                   # noqa: E402


def describe(backbone="1.5b"):
    print(f"{'ablation':20s} | {'name':34s} | {'delta':42s} | {'benchmarks':20s} | removes")
    for key, spec in ablations.ABLATIONS.items():
        delta = ", ".join(f"{k}={v}" for k, v in spec["delta"].items())
        bench = "+".join(ablations.benchmarks_for(key))
        print(f"{key:20s} | {spec['name']:34s} | {delta:42s} | {bench:20s} | {spec['removes']}")
    print("\nruns:")
    for key in ablations.ABLATIONS:
        for b in ablations.benchmarks_for(key):
            name, _ = ablations.build(key, b, backbone=backbone)
            print(f"  {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 epilog="extra arguments are forwarded to exp_run.py")
    ap.add_argument("--ablation", choices=sorted(ablations.ABLATIONS))
    ap.add_argument("--backbone", choices=sorted(arms.BACKBONE), default="1.5b")
    ap.add_argument("--list", action="store_true", help="print the ablation matrix and exit")
    arms.add_common_args(ap)
    if "--list" in sys.argv:
        listing = argparse.ArgumentParser(add_help=False)
        listing.add_argument("--backbone", choices=sorted(arms.BACKBONE), default="1.5b")
        selected, _ = listing.parse_known_args()
        describe(selected.backbone)
        return 0
    a, passthrough = ap.parse_known_args()
    if not a.ablation:
        ap.error("--ablation is required (see --list)")

    allowed = ablations.benchmarks_for(a.ablation)
    if a.benchmark not in allowed:
        ap.error(f"{a.ablation} is defined for {', '.join(allowed)} only "
                 f"(see --list); got --benchmark {a.benchmark}")
    name, cfg = ablations.build(a.ablation, a.benchmark, backbone=a.backbone)
    arms.check_gpus(a.gpus, a.backbone)
    spec = ablations.ABLATIONS[a.ablation]
    print(f"[arm] ablation   {spec['name']} -- {spec['title']}")
    print(f"[arm] removes    {spec['removes']}")
    print(f"[arm] control    {ablations.control_name(a.benchmark, a.ablation, backbone=a.backbone)}")
    return arms.launch(a.name or name, cfg, a.benchmark, a.gpus, passthrough,
                       allow_sdpa=a.allow_sdpa,
                       label=f"{spec['name']}  ({a.ablation}, {a.benchmark}, {a.backbone})")


if __name__ == "__main__":
    sys.exit(main())
