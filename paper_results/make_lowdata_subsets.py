#!/usr/bin/env python3
"""Build low-data subsets for the sparse-data sensitivity sweep (reviewer #5).

For each n in NS, keep each user's chronologically FIRST n sessions (prefix
truncation). A prefix keeps genuine consecutive pairs, so inter-arrival and
idle-gap structure stays real; random subsampling would inflate gaps and
corrupt the semi-Markov fit. Users with fewer than n sessions keep all.

Usage:  python make_lowdata_subsets.py [--csv PATH] [--out DIR] [--ns 10 20 30 50 100]
Writes: <out>/combined_first{n:03d}.csv
"""
import argparse, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=os.path.join(HERE, "dataset", "combined", "combined_sessions.csv"))
ap.add_argument("--out", default=os.path.join(HERE, "dataset", "lowdata"))
ap.add_argument("--ns", type=int, nargs="+", default=[10, 20, 30, 50, 100])
args = ap.parse_args()

df = pd.read_csv(args.csv)
df["abs_start"] = pd.to_datetime(df["abs_start"])
df = df.sort_values(["user_id", "abs_start"], kind="mergesort")
# Julia's DateTime parser needs the ISO 'T' separator; pandas writes a space.
df["abs_start"] = df["abs_start"].dt.strftime("%Y-%m-%dT%H:%M:%S")
os.makedirs(args.out, exist_ok=True)

full_counts = df.groupby("user_id").size()
print(f"source: {args.csv}")
print(f"  users={len(full_counts)}  sessions={len(df)}  "
      f"sessions/user min={full_counts.min()} med={int(full_counts.median())} max={full_counts.max()}")

for n in args.ns:
    sub = df.groupby("user_id", group_keys=False).head(n)
    out = os.path.join(args.out, f"combined_first{n:03d}.csv")
    # keep original column order / formats untouched apart from abs_start dtype
    sub.to_csv(out, index=False)
    c = sub.groupby("user_id").size()
    short = int((full_counts < n).sum())
    print(f"n={n:3d}: {len(c)} users, {len(sub)} sessions "
          f"({short} users have <{n} and keep all) -> {out}")
print("done. Next: for n in 10 20 30 50 100; do julia --project=. run_lowdata_N.jl $n; done")
