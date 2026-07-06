#!/usr/bin/env python3
# =============================================================================
# ACN-Data (Caltech workplace charging) -> SPARC 16-column schema
# =============================================================================
# ACN's per-DRIVER userID is very sparse (~1% of Caltech sessions), so per-user
# recurrence is not achievable. Instead we key on the CHARGING STATION: at a
# workplace, each charger is used repeatedly by its regular commuters, giving a
# recurring entity with plenty of sessions. This demonstrates SPARC on WORKPLACE
# per-charger charging (a valid generalization beyond residential per-household).
#
# Input priority: --json FILE  >  --token (fetch API)  >  existing checkpoint
#                 (dataset/acn_caltech/_acn_raw_<site>.jsonl, offline).
# Key:  --key stationID (default) | spaceID | userID
#
# Output: dataset/acn_caltech/acn_sessions.csv  ->  julia run_smc.jl acn
# =============================================================================
import argparse, json, os, sys, glob
import pandas as pd

N_MIN = 30         # min sessions to keep an entity
TOP_N = 15         # keep this many most-active entities (stations)
ENERGY_MIN, ENERGY_MAX = 0.5, 150.0
DUR_MIN_H, DUR_MAX_H   = 2/60, 120.0

MONTHS = ["January","February","March","April","May","June","July","August",
          "September","October","November","December"]
DAYS   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
HERE   = os.path.dirname(os.path.abspath(__file__))
PAGE   = 100


def fetch_api(token, site):
    import requests, time
    base = "https://ev.caltech.edu/api/v1/"
    raw  = os.path.join(HERE, "_acn_raw_%s.jsonl" % site)
    items = []
    if os.path.exists(raw):
        for line in open(raw):
            line = line.strip()
            if line: items.append(json.loads(line))
        print("resuming: %d sessions already on disk" % len(items))
    page = len(items) // PAGE + 1
    sess = requests.Session(); sess.auth = (token, '')
    out  = open(raw, "a")
    while True:
        url = base + "sessions/%s?max_results=%d&page=%d" % (site, PAGE, page)
        r = None
        for attempt in range(10):
            try:
                r = sess.get(url, timeout=60)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError("transient %s" % r.status_code)
                r.raise_for_status(); break
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                if attempt == 9:
                    out.flush(); out.close()
                    print("\nServer failing (%s). Saved %d sessions; re-run to resume from page %d."
                          % (e, len(items), page)); sys.exit(0)
                wait = min(2 ** attempt, 30)
                print("    %s -> retry %d/10 in %ds" % (e, attempt + 1, wait)); time.sleep(wait)
        batch = r.json().get('_items', [])
        if not batch: break
        for it in batch: out.write(json.dumps(it) + "\n")
        out.flush(); items += batch
        print("  fetched %d sessions (page %d)..." % (len(items), page))
        if not r.json().get('_links', {}).get('next'): break
        page += 1
    out.close(); return items


def load_checkpoint(site):
    fp = os.path.join(HERE, "_acn_raw_%s.jsonl" % site)
    if not os.path.exists(fp): return None
    items = [json.loads(l) for l in open(fp) if l.strip()]
    print("offline: loaded %d sessions from checkpoint %s" % (len(items), os.path.basename(fp)))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--token'); ap.add_argument('--site', default='caltech')
    ap.add_argument('--json'); ap.add_argument('--key', default='stationID',
                    choices=['stationID', 'spaceID', 'userID'])
    a = ap.parse_args()
    KEY = a.key

    if a.json:
        raw = json.load(open(a.json)); items = raw['_items'] if isinstance(raw, dict) else raw
    elif a.token:
        items = fetch_api(a.token, a.site)
    else:
        items = load_checkpoint(a.site)
        if items is None:
            raise SystemExit("no --json, no --token, and no checkpoint found")

    df = pd.DataFrame(items)
    if '_id' in df.columns: df = df.drop_duplicates('_id')
    df = df[df.get(KEY).notna()]
    df['t0'] = pd.to_datetime(df['connectionTime'], utc=True, errors='coerce')
    df['t1'] = pd.to_datetime(df['disconnectTime'], utc=True, errors='coerce')
    df['energy'] = pd.to_numeric(df['kWhDelivered'], errors='coerce')
    df = df.dropna(subset=['t0', 't1', 'energy', KEY])
    df['dur_h'] = (df['t1'] - df['t0']).dt.total_seconds() / 3600.0
    df = df[(df.energy > ENERGY_MIN) & (df.energy < ENERGY_MAX) &
            (df.dur_h >= DUR_MIN_H) & (df.dur_h <= DUR_MAX_H)]

    cnt  = df.groupby(KEY).size()
    keep = cnt[cnt >= N_MIN].sort_values(ascending=False).head(TOP_N).index
    df   = df[df[KEY].isin(keep)].sort_values([KEY, 't0']).reset_index(drop=True)
    print("kept %d sessions from %d entities (key=%s, >= %d sessions, top %d)"
          % (len(df), df[KEY].nunique(), KEY, N_MIN, TOP_N))

    ids = list(dict.fromkeys(df[KEY])); uidnum = {u: i for i, u in enumerate(ids)}
    t0 = df.t0.dt.tz_localize(None); t1 = df.t1.dt.tz_localize(None)
    out = pd.DataFrame({
        'location':         ['US_workplace'] * len(df),
        'user_id':          ['ACN_' + str(u) for u in df[KEY]],
        'session_id':       ['ACN' + str(i) for i in range(1, len(df) + 1)],
        'duration_hours':   df.dur_h.values,
        'energy':           df.energy.values,
        'start_date':       t0.dt.strftime('%Y-%m-%d'), 'start_time': t0.dt.strftime('%H:%M:%S'),
        'end_date':         t1.dt.strftime('%Y-%m-%d'), 'end_time':   t1.dt.strftime('%H:%M:%S'),
        'start_hour':       t0.dt.hour, 'end_hour': t1.dt.hour,
        'month':            [MONTHS[mo - 1] for mo in t0.dt.month],
        'day':              [DAYS[w] for w in t0.dt.weekday],
        'duration_minutes': df.dur_h.values * 60,
        'day_type':         ['Weekend' if w >= 5 else 'Weekday' for w in t0.dt.weekday],
        'user_id_num':      [uidnum[u] for u in df[KEY]],
    })
    fp = os.path.join(HERE, 'acn_sessions.csv'); out.to_csv(fp, index=False)
    spc = out.groupby('user_id').size()
    print("\nSaved -> %s" % fp)
    print("FINAL: %d sessions, %d entities (median %d, range %d-%d sessions/entity)"
          % (len(out), out.user_id.nunique(), int(spc.median()), spc.min(), spc.max()))


if __name__ == '__main__':
    main()
