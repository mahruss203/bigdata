import pandas as pd
from darts import TimeSeries
from darts.models import NBEATSModel
import torch
from typing import Dict
import matplotlib.pyplot as plt
from darts.metrics import mape, mae, rmse
import os

# Disable MPI detection to avoid errors
os.environ['MPI4PY_RC_INITIALIZE'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

print("=" * 60)
print("STEP 1: PREPROCESSING DATA")
print("=" * 60)

# Membaca data
df = pd.read_csv("datatime.csv")
print("\n✓ Data berhasil dibaca dari datatime.csv")
print(f"  Shape: {df.shape}")

# Mengecek missing value per kolom
missing_per_column = df.isnull().sum()
print("\nMissing values per kolom:")
print(missing_per_column)

# Hapus baris entitas agregat (bukan negara)
REMOVE = [
    "World", "Asia", "Africa", "Europe", "European Union",
    "North America", "South America", "Latin America and Caribbean",
    "Oceania"
]

df = df[~df['Country'].isin(REMOVE)]
print(f"\n✓ Entitas agregat dihapus. Shape sekarang: {df.shape}")

# Hapus duplikasi jika ada
df = df.drop_duplicates()

# Tangani missing value dengan interpolasi per negara
df['Stroke_Deaths'] = df.groupby('Country')['Stroke_Deaths'] \
                        .transform(lambda x: x.interpolate(method='linear'))

# Jika masih ada NA → isi median negara
df['Stroke_Deaths'] = df.groupby('Country')['Stroke_Deaths'] \
                        .transform(lambda x: x.fillna(x.median()))

# Ubah tipe data agar rapi
df['Year'] = df['Year'].astype(int)
df['Stroke_Deaths'] = df['Stroke_Deaths'].astype(float)

# Sort berdasarkan negara dan tahun
df = df.sort_values(by=['Country', 'Year']).reset_index(drop=True)

# Hapus kolom 'Code'
df = df.drop(columns=['Code'])

# Simpan hasil ke file CSV baru
df.to_csv("datafix.csv", index=False)
print("✓ Data preprocessing selesai, disimpan ke datafix.csv")

print("\n" + "=" * 60)
print("STEP 2: KONVERSI KE TIME SERIES")
print("=" * 60)

# Ubah Year jadi datetime
df["Year"] = pd.to_datetime(df["Year"].astype(str), format="%Y")
print("\n✓ Kolom Year dikonversi ke datetime")

# Pisahkan data per negara
country_dfs = {
    country: sub_df[["Year", "Stroke_Deaths"]].reset_index(drop=True)
    for country, sub_df in df.groupby("Country")
}
print(f"✓ Data dipecah menjadi {len(country_dfs)} negara")

# Ubah ke TimeSeries per negara
series_by_country: Dict[str, TimeSeries] = {}

for country, cdf in country_dfs.items():
    ts = TimeSeries.from_dataframe(
        cdf,
        time_col="Year",
        value_cols="Stroke_Deaths"
    )
    series_by_country[country] = ts

print(f"✓ {len(series_by_country)} TimeSeries berhasil dibuat")

print("\n" + "=" * 60)
print("STEP 3: TRAINING MODEL N-BEATS")
print("=" * 60)

input_chunk_length = 2
output_chunk_length = 1
min_required_len = input_chunk_length + output_chunk_length

print(f"\nParameter model:")
print(f"  - Input chunk length: {input_chunk_length}")
print(f"  - Output chunk length: {output_chunk_length}")
print(f"  - Minimal panjang data: {min_required_len}")

train_series = []

for country, ts in series_by_country.items():
    L = len(ts)
    if L < min_required_len:
        continue
    train_series.append(ts)

print(f"\n✓ {len(train_series)} negara memenuhi syarat untuk training")

# MODEL GLOBAL
global_model = NBEATSModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=output_chunk_length,
    n_epochs=5,
    batch_size=16,
    random_state=42,
    pl_trainer_kwargs={"accelerator": "cpu"}
)

print("\n🔥 Mulai training model global N-BEATS...")
global_model.fit(series=train_series, verbose=True)
print("\n✓ Training model global selesai!")

print("\n" + "=" * 60)
print("STEP 4: EVALUASI MODEL")
print("=" * 60)

def ts_to_df(ts):
    """Konversi Darts TimeSeries ke pandas DataFrame."""
    if hasattr(ts, "to_dataframe"):
        df = ts.to_dataframe().reset_index()
    else:
        raise AttributeError("TimeSeries tidak punya to_dataframe().")
    
    cols = list(df.columns)
    cols[0] = "Date"
    df.columns = cols
    return df

eval_rows = []

print("\n📊 Mengevaluasi model per negara...")

for country, ts in series_by_country.items():
    L = len(ts)
    
    if L < min_required_len + 1:
        continue
    
    try:
        backtest = global_model.historical_forecasts(
            ts,
            start=0.5,
            forecast_horizon=1,
            stride=1,
            retrain=False,
            last_points_only=True,
            verbose=False,
        )
        
        from darts import TimeSeries
        if isinstance(backtest, list):
            if len(backtest) == 0:
                continue
            backtest = backtest[0]
        
        if not isinstance(backtest, TimeSeries):
            continue
        
        actual = ts.slice_intersect(backtest)
        backtest = backtest.slice_intersect(actual)
        
        if len(actual) == 0 or len(backtest) == 0:
            continue
        
        mape_val = mape(actual, backtest)
        mae_val  = mae(actual, backtest)
        rmse_val = rmse(actual, backtest)
        
        eval_rows.append({
            "Country": country,
            "MAPE": mape_val,
            "MAE": mae_val,
            "RMSE": rmse_val,
        })
    except Exception as e:
        continue

eval_df = pd.DataFrame(eval_rows).sort_values("MAPE").reset_index(drop=True)

print("\n✓ Evaluasi selesai!")
print(f"\nTop 10 negara dengan MAPE terbaik:")
print(eval_df.head(10).to_string(index=False))

print(f"\n📈 Statistik evaluasi:")
print(f"  - Rata-rata MAPE: {eval_df['MAPE'].mean():.2f}%")
print(f"  - Rata-rata MAE: {eval_df['MAE'].mean():.2f}")
print(f"  - Rata-rata RMSE: {eval_df['RMSE'].mean():.2f}")

print("\n" + "=" * 60)
print("STEP 5: MENYIMPAN MODEL DAN HASIL")
print("=" * 60)

# Simpan model
global_model.save("nbeats_global_model.pth")
print("\n✓ Model N-BEATS disimpan: nbeats_global_model.pth")

# Simpan hasil evaluasi
eval_df.to_csv("hasil_evaluasi_nbeats.csv", index=False)
print("✓ Hasil evaluasi disimpan: hasil_evaluasi_nbeats.csv")

print("\n" + "=" * 60)
print("✅ PROSES SELESAI!")
print("=" * 60)
print("\nFile yang dihasilkan:")
print("  1. datafix.csv - Data yang sudah dibersihkan")
print("  2. nbeats_global_model.pth - Model N-BEATS")
print("  3. hasil_evaluasi_nbeats.csv - Hasil evaluasi per negara")
