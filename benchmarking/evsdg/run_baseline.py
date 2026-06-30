"""
One-command EV-SDG baseline runner (combined dataset).

Runs the three EV-SDG stages on res/transactions.csv and copies the synthetic
sessions to results/EVSDG_combined_generated.csv.

    python benchmarking/evsdg/run_baseline.py

The baseline output is already committed in results/; re-run only to reproduce.
Requires the deps in requirements.txt (pandas<2 — EV-SDG uses DataFrame.append).
"""
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # benchmarking/evsdg/


def run(script, *args):
    subprocess.run([sys.executable, script, *args], cwd=HERE, check=True)


def main():
    run('SDG_preprocessing.py', '-Year', '2019', '-Slotmins', '60',
        '-Sessions_filename', 'transactions.csv', '-res_folder', 'res', '-verbose', '0')
    run('SDG_fit.py', '-model', 'IAT', '-lambdamod', 'mean', '-verbose', '0')
    run('SDG_sample_generate.py', '-start_date', '01/01/2019', '-end_date', '31/12/2019',
        '-use', 'latest', '-model', 'IAT', '-lambdamod', 'mean', '-verbose', '0')

    gens = sorted(glob.glob(str(HERE / 'res' / 'generated_samples' / '*.csv')),
                  key=os.path.getmtime)
    (HERE / 'results').mkdir(exist_ok=True)
    out = HERE / 'results' / 'EVSDG_combined_generated.csv'
    shutil.copy(gens[-1], out)
    print('baseline synthetic sessions ->', out)


if __name__ == '__main__':
    main()
