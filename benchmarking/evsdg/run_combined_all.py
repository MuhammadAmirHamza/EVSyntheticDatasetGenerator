"""EV-SDG baseline on the SPARC combined pool (170 entities), one command.

    python benchmarking/evsdg/run_combined_all.py [path/to/raw_clean.csv]

Steps: (1) make_transactions.py builds res/transactions.csv from the cleaned
SPARC cohort, (2) the three EV-SDG stages run on it, (3) the generated sessions
are written to results/EVSDG_combined_all.csv with the columns
arrival_hour, duration_h, energy_kwh used by the benchmarking scripts.
Needs the packages in requirements.txt (pandas < 2.0, Python 3.10 or 3.11).
"""
import glob, os, subprocess, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent


def run(script, *a):
    subprocess.run([sys.executable, '-W', 'ignore', script, *a], cwd=HERE, check=True)


def main():
    run('make_transactions.py', *sys.argv[1:2])
    for f in glob.glob(str(HERE / 'res' / 'generated_samples' / '*.csv')):
        os.remove(f)
    run('SDG_preprocessing.py', '-Year', '2019', '-Slotmins', '60',
        '-Sessions_filename', 'transactions.csv', '-res_folder', 'res', '-verbose', '0')
    run('SDG_fit.py', '-model', 'IAT', '-lambdamod', 'mean', '-verbose', '0')
    run('SDG_sample_generate.py', '-start_date', '01/01/2019', '-end_date', '31/12/2019',
        '-use', 'latest', '-model', 'IAT', '-lambdamod', 'mean', '-verbose', '0')
    gens = sorted(glob.glob(str(HERE / 'res' / 'generated_samples' / '*.csv')), key=os.path.getmtime)
    if not gens:
        sys.exit('EV-SDG produced no sample file')
    d = pd.read_csv(gens[-1])
    out = pd.DataFrame({'arrival_hour': d.Arrival, 'duration_h': d.Connected_time,
                        'energy_kwh': d.Energy_required})
    out = out[(out.duration_h > 0) & (out.energy_kwh > 0)]
    (HERE / 'results').mkdir(exist_ok=True)
    dst = HERE / 'results' / 'EVSDG_combined_all.csv'
    out.to_csv(dst, index=False)
    print(f'{len(out)} synthetic sessions -> {dst}')


if __name__ == '__main__':
    main()
