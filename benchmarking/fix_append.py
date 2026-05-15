"""Replace deprecated DataFrame.append with pd.concat in EV-SDG source.

Pattern: X = X.append(EXPR)  →  X = pd.concat([X, EXPR], ignore_index=True)
For .append([scalar]) cases, EXPR becomes pd.DataFrame([scalar]).
"""
import re
from pathlib import Path

files = [
    'benchmarking/EV-SDG-master/modeling/stat/exponential_process.py',
    'benchmarking/EV-SDG-master/modeling/stat/poisson_process.py',
    'benchmarking/EV-SDG-master/modeling/generate_sample.py',
]

pat_df = re.compile(r'(\b\w+)\s*=\s*\1\.append\((pd\.DataFrame\([^)]+\))\)')
pat_list = re.compile(r'(\btime_slot)\s*=\s*\1\.append\(\[([^\]]+)\]\)')
pat_other = re.compile(r'(\b\w+)\s*=\s*\1\.append\((\w+)\)')

for fp in files:
    p = Path(fp)
    text = p.read_text()
    new = text
    new = pat_df.sub(r'\1 = pd.concat([\1, \2], ignore_index=True)', new)
    new = pat_list.sub(r'\1 = pd.concat([\1, pd.DataFrame([\2])], ignore_index=True)', new)
    new = pat_other.sub(r'\1 = pd.concat([\1, \2], ignore_index=True)', new)
    if new != text:
        p.write_text(new)
        n = text.count('.append(') - new.count('.append(')
        print(f'  {fp}: replaced {n} occurrence(s)')
    else:
        print(f'  {fp}: no changes')
