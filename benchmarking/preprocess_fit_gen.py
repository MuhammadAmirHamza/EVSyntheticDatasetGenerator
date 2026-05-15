"""End-to-end EV-SDG pipeline runner.

Runs all four stages in sequence:
    1. Convert raw_clean.csv -> EV-SDG-master/res/transactions.csv (4-column format)
    2. Preprocess (SDG_preprocessing.py): slot data + session/pole clusters
    3. Fit AM + MMc + MMe models (SDG_fit.py)
    4. Generate synthetic samples (SDG_sample_generate.py)

Usage (run with the venv's python from the project root):

    benchmarking\\EV-SDG-master\\benchmarking\\EV-SDG-master\\.venv\\Scripts\\python.exe ^
        benchmarking\\preprocess_fit_gen.py

Optional flags:
    --year 2019              year to preprocess and train on
    --slot-mins 60           slot length in minutes
    --model IAT|AC           arrival model family (default IAT)
    --lambdamod mean|loess|poly|poisson_fit|neg_bio_reg
    --start dd/mm/YYYY       generation horizon start (default 01/01/2019)
    --end   dd/mm/YYYY       generation horizon end   (default 31/01/2019)
    --verbose 0..3           verbosity for each SDG step
    --skip 1 2 3 4           skip listed steps (e.g. --skip 1 2 to only fit + generate)
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(r'D:\codes\probabilitistic codes\SMC files\v2\benchmarking').resolve()           # benchmarking/
SDG_DIR = HERE / 'EV-SDG-master'                 # benchmarking/EV-SDG-master
RES_DIR = SDG_DIR / 'res'
RAW = HERE / 'raw_clean.csv'


def step_1_convert():
    import pandas as pd
    df = pd.read_csv(RAW)
    out = pd.DataFrame({
        'Started': pd.to_datetime(df['abs_start']).dt.strftime('%d/%m/%Y %H:%M:%S'),
        'ConnectedTime': df['duration_hours'].astype(float),
        'TotalEnergy': df['energy'].astype(float),
        'ChargePoint': df['user_id'].astype(str),
    })
    RES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RES_DIR / 'transactions.csv', index=False)
    print(f'[1/4] transactions.csv written: {len(out)} rows, '
          f'{out.ChargePoint.nunique()} charge points')


def run_sdg(script, *args):
    subprocess.run([sys.executable, script, *args], cwd=SDG_DIR, check=True)


def step_2_preprocess(year, slot_mins, verbose):
    print(f'[2/4] Preprocessing year={year}, slot={slot_mins}min')
    run_sdg('SDG_preprocessing.py',
            '-Year', str(year),
            '-Slotmins', str(slot_mins),
            '-Sessions_filename', 'transactions.csv',
            '-res_folder', 'res',
            '-verbose', str(verbose))


def step_3_fit(model, lambdamod, verbose):
    print(f'[3/4] Fitting {model}/{lambdamod}')
    run_sdg('SDG_fit.py',
            '-model', model,
            '-lambdamod', lambdamod,
            '-verbose', str(verbose))


def step_4_generate(start, end, model, lambdamod, verbose):
    print(f'[4/4] Generating samples {start} -> {end}')
    run_sdg('SDG_sample_generate.py',
            '-start_date', start,
            '-end_date', end,
            '-use', 'latest',
            '-model', model,
            '-lambdamod', lambdamod,
            '-verbose', str(verbose))


def main():
    p = argparse.ArgumentParser(description='End-to-end EV-SDG pipeline')
    p.add_argument('--year', type=int, default=2019)
    p.add_argument('--slot-mins', type=int, default=60)
    p.add_argument('--model', default='IAT', choices=['IAT', 'AC'])
    p.add_argument('--lambdamod', default='mean')
    p.add_argument('--start', default='01/01/2019', help='Generation start dd/mm/YYYY')
    p.add_argument('--end', default='31/01/2023', help='Generation end dd/mm/YYYY')
    p.add_argument('--verbose', type=int, default=1)
    p.add_argument('--skip', nargs='*', default=[], choices=['1', '2', '3', '4'],
                   help='Skip steps by number')
    args = p.parse_args()

    if '1' not in args.skip:
        step_1_convert()
    if '2' not in args.skip:
        step_2_preprocess(args.year, args.slot_mins, args.verbose)
    if '3' not in args.skip:
        step_3_fit(args.model, args.lambdamod, args.verbose)
    if '4' not in args.skip:
        step_4_generate(args.start, args.end, args.model, args.lambdamod, args.verbose)

    print('\nPipeline complete.')
    print(f'  Preprocessed data: {RES_DIR / "preprocess"}')
    print(f'  Fitted models:     {RES_DIR / "models" / "saved_models"}')
    print(f'  Generated samples: {RES_DIR / "generated_samples"}')


if __name__ == '__main__':
    main()
