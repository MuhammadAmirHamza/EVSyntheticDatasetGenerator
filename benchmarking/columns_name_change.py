import pandas as pd

df = pd.read_csv('benchmarking/raw_clean.csv')

out = pd.DataFrame({
    'Started': pd.to_datetime(df['abs_start']).dt.strftime('%d/%m/%Y %H:%M:%S'),
    'ConnectedTime': df['duration_hours'].astype(float),
    'TotalEnergy': df['energy'].astype(float),
    'ChargePoint': df['user_id'].astype(str),
})

## move the file in res folder
import os
out_dir = 'benchmarking/EV-SDG-master/res'
os.makedirs(out_dir, exist_ok=True)
out.to_csv(os.path.join(out_dir, 'transactions.csv'), index=False)
print(out.head())

