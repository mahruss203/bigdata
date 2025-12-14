"""
Script untuk Pre-compute ARIMA Models untuk Semua Negara
Hasil disimpan dalam cache untuk akses cepat di web app
"""

import pandas as pd
import numpy as np
from pmdarima import auto_arima
import pickle
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

def prepare_data():
    """Membaca dan mempersiapkan data"""
    print("📊 Membaca data timeseries...")
    df = pd.read_csv('timeseries.csv')
    
    # Rename kolom
    df = df.rename(columns={
        'Entity': 'Country',
        'Total deaths from stroke among both sexes': 'Stroke_Deaths'
    })
    
    # Hapus kolom Code jika ada
    if 'Code' in df.columns:
        df = df.drop(columns=['Code'])
    
    # Hapus entitas agregat
    REMOVE = [
        "World", "Asia", "Africa", "Europe", "European Union",
        "North America", "South America", "Latin America and Caribbean",
        "Oceania"
    ]
    df = df[~df['Country'].isin(REMOVE)]
    
    # Preprocessing
    df = df.drop_duplicates()
    df['Stroke_Deaths'] = df.groupby('Country')['Stroke_Deaths'].transform(
        lambda x: x.interpolate(method='linear')
    )
    df['Stroke_Deaths'] = df.groupby('Country')['Stroke_Deaths'].transform(
        lambda x: x.fillna(x.median())
    )
    df['Year'] = df['Year'].astype(int)
    df['Stroke_Deaths'] = df['Stroke_Deaths'].astype(float)
    df = df.sort_values(by=['Country', 'Year']).reset_index(drop=True)
    
    return df

def train_arima_for_country(country, ts_data):
    """Train ARIMA model untuk satu negara"""
    try:
        # Fit auto_arima
        model = auto_arima(
            ts_data,
            seasonal=False,
            suppress_warnings=True,
            stepwise=True,
            error_action='ignore',
            max_p=5,
            max_q=5,
            max_d=2
        )
        
        # Simpan order dan fitted model
        return {
            'model': model,
            'order': model.order,
            'last_year': ts_data.index.year.max(),
            'last_value': ts_data.values[-1],
            'status': 'success'
        }
    except Exception as e:
        return {
            'model': None,
            'order': None,
            'last_year': None,
            'last_value': None,
            'status': 'failed',
            'error': str(e)
        }

def generate_arima_cache():
    """Generate cache untuk semua negara"""
    print("🚀 Memulai proses pre-compute ARIMA models...\n")
    
    # Load data
    df = prepare_data()
    list_negara = sorted(df['Country'].unique())
    
    print(f"📌 Total negara: {len(list_negara)}")
    print(f"📅 Range tahun: {df['Year'].min()} - {df['Year'].max()}\n")
    
    # Dictionary untuk menyimpan models
    arima_cache = {}
    failed_countries = []
    
    # Loop untuk setiap negara dengan progress bar
    print("⏳ Training ARIMA models untuk semua negara...\n")
    for country in tqdm(list_negara, desc="Progress"):
        # Filter data untuk negara
        country_data = df[df['Country'] == country].copy()
        ts = country_data.groupby('Year')['Stroke_Deaths'].sum()
        ts.index = pd.to_datetime(ts.index, format='%Y')
        
        # Train ARIMA
        result = train_arima_for_country(country, ts)
        
        if result['status'] == 'success':
            arima_cache[country] = result
        else:
            failed_countries.append(country)
            print(f"\n⚠️  Failed: {country} - {result['error']}")
    
    # Simpan cache
    print("\n💾 Menyimpan ARIMA cache...")
    with open('arima_models_cache.pkl', 'wb') as f:
        pickle.dump(arima_cache, f)
    
    # Simpan metadata
    metadata = {
        'total_countries': len(list_negara),
        'successful': len(arima_cache),
        'failed': len(failed_countries),
        'failed_countries': failed_countries,
        'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('arima_cache_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    # Summary
    print("\n" + "="*60)
    print("✅ ARIMA Cache Generation SELESAI!")
    print("="*60)
    print(f"✔️  Berhasil: {metadata['successful']} negara")
    print(f"❌ Gagal: {metadata['failed']} negara")
    if failed_countries:
        print(f"   Negara gagal: {', '.join(failed_countries)}")
    print(f"📁 File cache: arima_models_cache.pkl")
    print(f"📁 Metadata: arima_cache_metadata.pkl")
    print("="*60)
    
    return arima_cache, metadata

if __name__ == '__main__':
    generate_arima_cache()
