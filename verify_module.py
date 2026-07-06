"""
Verify the shared-core module reproduces the per-dataset scripts.

For each dataset it MD5-compares the key outputs of the original run
(artifacts/<name>/) against the module run (artifacts/<name>_module/).
Because the framework calls Random.seed!(24), a faithful refactor yields
byte-identical files, so every row below should read "OK".

Workflow (per dataset, e.g. norway):
    julia SMC_framework_norway.jl        # original  -> artifacts/norway_sorensen/
    julia run_smc.jl norway              # module    -> artifacts/norway_sorensen_module/
    python verify_module.py norway       # compare

    python verify_module.py              # compare all four
"""
import sys, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = {'norway': 'norway_sorensen', 'korea': 'korea_gist',
         'pecan': 'pecanstreet', 'combined': 'combined'}
KEY = ['kl_divergence.txt', 'validation_l1.csv', 'validation_l2.csv',
       'validation_l3.csv', 'validation_l4.csv', 'sim_sessions.csv',
       'profiles_summary.csv', 'pairs.csv', 'raw_clean.csv',
       'convolution_diagnostic.csv']


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None


def main():
    datasets = sys.argv[1:] or list(NAMES)
    overall = True
    for ds in datasets:
        if ds not in NAMES:
            print(f"unknown dataset {ds!r}; choose from {list(NAMES)}"); continue
        name = NAMES[ds]
        a = ROOT / 'artifacts' / name
        b = ROOT / 'artifacts' / (name + '_module')
        print(f"\n=== {ds}:  {a.name}/  vs  {b.name}/ ===")
        if not b.exists():
            print(f"  module output missing — run:  julia run_smc.jl {ds}")
            overall = False; continue
        if not a.exists():
            print(f"  original output missing — run the original SMC_framework_*.jl")
            overall = False; continue
        allok = True
        for f in KEY:
            ha, hb = md5(a / f), md5(b / f)
            if ha is None and hb is None:
                continue
            ok = (ha == hb)
            allok &= ok
            tag = 'OK  ' if ok else 'DIFF'
            print(f"  {tag}  {f}")
        print("  => IDENTICAL ✓" if allok else "  => DIFFERENCES FOUND ✗")
        overall &= allok
    print("\n" + ("ALL DATASETS IDENTICAL ✓" if overall else
                  "some differences / missing runs — see above"))


if __name__ == '__main__':
    main()
