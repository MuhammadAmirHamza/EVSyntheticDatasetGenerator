#!/usr/bin/env python3
# =============================================================================
# Pecan Street (Dataport, free University tier) — STATIC downloads → sessions
# =============================================================================
# Reads the 1-minute regional archives downloaded from Dataport's "Static
# Downloads" (Austin / California / New York / Puerto Rico .tar.gz), which sit
# in THIS folder, keeps ONLY the EV-charging circuit (car1/car2), reconstructs
# charging sessions, removes outliers, and writes them in the SAME 16-column
# schema as the Norway/Korea sets so SMC_framework ingests it with no changes.
#
#   Input : dataset/pecanstreet/*.tar.gz   (each holds <region>/<region>.csv)
#   Output: dataset/pecanstreet/pecanstreet_sessions.csv
#
# Run:  pip install pandas
#       python dataset\pecanstreet\pecanstreet_preprocess.py
# (No database / no credentials — the paid DB tier is not needed.)
# =============================================================================

import os, glob, tarfile
import pandas as pd

# ----- parameters -------------------------------------------------------------
CHARGE_KW      = 0.3          # car1+car2 above this (kW) = actively charging
GAP_SPLIT_MIN  = 180         # DISCONNECT gap: a plug-in ends only after >this many minutes
                             #   with NO charging. Bridges full-battery lulls / intermittent
                             #   top-ups into one plug-in; real disconnects run ~8-24 h. Tunable.
ENERGY_MIN_KWH = 0.5         # drop aborted / near-zero sessions
ENERGY_MAX_KWH = 150.0       # sanity cap
DUR_MIN_HOURS  = 2/60        # 2 minutes
DUR_MAX_HOURS  = 120.0       # 5 days
CHUNK          = 500_000     # rows per read chunk (keeps memory bounded)
WANT           = {"dataid", "localminute", "car1", "car2"}
EXCLUDE_HOMES  = {661}        # dataids to drop (PS_661: atypical 3-4 a.m. timer, non-representative)

# ----- paths (this script lives in dataset/pecanstreet/, with the archives) ----
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
DATA_DIR = HERE
OUT      = os.path.join(DATA_DIR, "pecanstreet_sessions.csv")

MONTHS = ["January","February","March","April","May","June","July","August",
          "September","October","November","December"]
DAYS   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

def main_csv_member(tf):
    """The big per-region data CSV inside the archive (not the metadata.csv)."""
    csvs = [m for m in tf.getmembers()
            if m.name.lower().endswith(".csv") and "metadata" not in m.name.lower()]
    return max(csvs, key=lambda m: m.size) if csvs else None

# ----- 1. Read EV-charging minutes from every regional archive ----------------
parts = []
for targz in sorted(glob.glob(os.path.join(DATA_DIR, "*.tar.gz"))):
    region = os.path.basename(targz)
    with tarfile.open(targz, "r:gz") as tf:
        member = main_csv_member(tf)
        if member is None:
            print(f"  {region}: no data CSV found, skipping"); continue
        f = tf.extractfile(member)
        kept = 0
        for chunk in pd.read_csv(f, usecols=lambda c: c in WANT,
                                 chunksize=CHUNK, low_memory=False):
            c1 = pd.to_numeric(chunk["car1"], errors="coerce").fillna(0) if "car1" in chunk else 0.0
            c2 = pd.to_numeric(chunk["car2"], errors="coerce").fillna(0) if "car2" in chunk else 0.0
            chunk["kw"] = c1 + c2
            keep = chunk.loc[chunk.kw > CHARGE_KW, ["dataid", "localminute", "kw"]]
            if len(keep):
                parts.append(keep); kept += len(keep)
        print(f"  {region}: {kept:,} EV-charging minutes")

if not parts:
    raise SystemExit("No EV-charging minutes found. Are the *.tar.gz archives in "
                     "this folder, and do any homes have a car1/car2 circuit?")

df = pd.concat(parts, ignore_index=True)
df = df[~df.dataid.isin(EXCLUDE_HOMES)]          # schedule-based curation (see EXCLUDE_HOMES)
# local wall-clock time (strip the trailing tz offset, e.g. '...-06')
df["ts"] = pd.to_datetime(df.localminute.str[:19], errors="coerce")
df = df.dropna(subset=["ts"])
print(f"total EV-charging minutes: {len(df):,} across {df.dataid.nunique()} homes")

# ----- 2. Reconstruct sessions: contiguous charging minutes per home ----------
rows = []
for dataid, g in df.sort_values(["dataid", "ts"]).groupby("dataid"):
    g = g.reset_index(drop=True)
    gap = g.ts.diff().dt.total_seconds().div(60).fillna(0)   # minutes since prev
    sid = (gap > GAP_SPLIT_MIN).cumsum()
    for _, s in g.groupby(sid):
        t0 = s.ts.iloc[0]
        t1 = s.ts.iloc[-1] + pd.Timedelta(minutes=1)         # each row = 1-min slice
        rows.append((dataid, t0, t1, (t1 - t0).total_seconds()/3600, s.kw.sum()/60.0))
ses = pd.DataFrame(rows, columns=["dataid", "ts", "te", "dur_h", "energy"])
print(f"reconstructed {len(ses):,} raw sessions")

# ----- 3. Outlier removal -----------------------------------------------------
m = ((ses.energy > ENERGY_MIN_KWH) & (ses.energy < ENERGY_MAX_KWH) &
     (ses.dur_h >= DUR_MIN_HOURS)  & (ses.dur_h <= DUR_MAX_HOURS))
ses = ses[m].reset_index(drop=True)
print(f"after outlier removal: {len(ses):,} sessions, {ses.dataid.nunique()} homes")

# ----- 4. Build 16-column schema (matches norway_sorensen_sessions.csv) --------
ids = list(dict.fromkeys(ses.dataid)); uidnum = {u: i for i, u in enumerate(ids)}
out = pd.DataFrame({
    "location":         ["US_residential"] * len(ses),
    "user_id":          ["PS_" + str(u) for u in ses.dataid],
    "session_id":       ["PS" + str(i) for i in range(1, len(ses) + 1)],
    "duration_hours":   ses.dur_h.values,
    "energy":           ses.energy.values,
    "start_date":       ses.ts.dt.strftime("%Y-%m-%d"),
    "start_time":       ses.ts.dt.strftime("%H:%M:%S"),
    "end_date":         ses.te.dt.strftime("%Y-%m-%d"),
    "end_time":         ses.te.dt.strftime("%H:%M:%S"),
    "start_hour":       ses.ts.dt.hour,
    "end_hour":         ses.te.dt.hour,
    "month":            [MONTHS[mn-1] for mn in ses.ts.dt.month],
    "day":              [DAYS[w] for w in ses.ts.dt.weekday],
    "duration_minutes": ses.dur_h.values * 60,
    "day_type":         ["Weekend" if w >= 5 else "Weekday" for w in ses.ts.dt.weekday],
    "user_id_num":      [uidnum[u] for u in ses.dataid],
})
out.to_csv(OUT, index=False)
spc = out.groupby("user_id").size()
print(f"\nSaved → dataset/pecanstreet/pecanstreet_sessions.csv")
print(f"FINAL: {len(out):,} sessions, {out.user_id.nunique()} EV homes "
      f"(median {int(spc.median())}, range {spc.min()}–{spc.max()} sessions/home)")
print(f"  median duration {out.duration_hours.median():.2f} h | "
      f"median energy {out.energy.median():.2f} kWh")
