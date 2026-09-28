from __future__ import annotations
import argparse
from devspectral.experiment import run_one
from devspectral.visualize import render_snapshot, LAYERS


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--history',choices=['H_A','H_B'],default='H_A')
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--layer',choices=LAYERS,default='graph')
    ap.add_argument('--output')
    args=ap.parse_args()
    record=run_one(args.history,args.seed)
    fig=render_snapshot(record,args.layer,args.output)
    if args.output is None:
        import matplotlib.pyplot as plt
        plt.show()
    return 0

if __name__=='__main__': raise SystemExit(main())
