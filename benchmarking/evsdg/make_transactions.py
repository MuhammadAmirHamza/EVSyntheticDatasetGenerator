"""Build res/transactions.csv (EV-SDG input) from the cleaned SPARC cohort.

    python make_transactions.py [path/to/raw_clean.csv]

Default input: ../../paper_results/artifacts/combined_all_N200_b24_s01/raw_clean.csv
(the 170 entities / 31,536 sessions used by SPARC).  EV-SDG models one calendar
year, so every start date is moved into 2019 by a whole number of weeks
(multiples of 364 days): the weekday and the time of day are unchanged.
"""
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
default = HERE.parent.parent / 'paper_results' / 'artifacts' / 'combined_all_N200_b24_s01' / 'raw_clean.csv'
src = Path(sys.argv[1]) if len(sys.argv) > 1 else default

d = pd.read_csv(src)
start = pd.to_datetime(d['abs_start'])
lo, hi = pd.Timestamp('2019-01-01'), pd.Timestamp('2020-01-01')
shift = pd.Timedelta(days=364)
for _ in range(20):                                   # move into 2019, week-aligned
    start = start.where(start >= lo, start + shift)
    start = start.where(start < hi, start - shift)
ok = (start >= lo) & (start < hi)
out = pd.DataFrame({
    'Started':       start.dt.strftime('%d/%m/%Y %H:%M:%S'),
    'ConnectedTime': d['duration_hours'].astype(float),
    'TotalEnergy':   d['energy'].astype(float),
    'ChargePoint':   d['user_id'].astype(str),
})[ok]
(HERE / 'res').mkdir(exist_ok=True)
out.to_csv(HERE / 'res' / 'transactions.csv', index=False)
print(f'{len(out)} sessions, {out.ChargePoint.nunique()} charge points -> res/transactions.csv '
      f'({(~ok).sum()} could not be placed in 2019)')
