#!/bin/bash
ds=$1; yr=$2; PY=.venv/bin/python
cp inputs/transactions_$ds.csv res/transactions.csv; rm -f res/generated_samples/*.csv
$PY SDG_preprocessing.py -Year $yr -Slotmins 60 -Sessions_filename transactions.csv -res_folder res -verbose 0 >/dev/null 2>&1
$PY SDG_fit.py -model IAT -lambdamod mean -verbose 0 >/dev/null 2>&1
$PY SDG_sample_generate.py -start_date 01/01/$yr -end_date 31/12/$yr -use latest -model IAT -lambdamod mean -verbose 0 >/dev/null 2>&1
$PY - "$ds" <<'PY'
import pandas as pd, glob, sys
g=sorted(glob.glob("res/generated_samples/*.csv"))
if not g: print("  NO OUTPUT"); sys.exit(1)
d=pd.read_csv(g[-1]); out=pd.DataFrame({"arrival_hour":d.Arrival,"duration_h":d.Connected_time,"energy_kwh":d.Energy_required})
out=out[(out.duration_h>0)&(out.energy_kwh>0)]; out.to_csv(f"results/EVSDG_{sys.argv[1]}.csv",index=False)
print(f"  saved EVSDG_{sys.argv[1]}.csv ({len(out)} sessions)")
PY
