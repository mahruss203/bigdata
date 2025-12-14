import pandas as pd
import pickle
from darts import TimeSeries
from darts.models import NBEATSModel
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("GENERATING PRE-COMPUTED N-BEATS PREDICTIONS")
print("="*60)

# Baca data
df = pd.read_csv('datafix.csv')
df["Year"] = pd.to_datetime(df["Year"].astype(str), format="%Y")

# Konversi ke TimeSeries per negara
country_dfs = {
    country: sub_df[["Year", "Stroke_Deaths"]].reset_index(drop=True)
    for country, sub_df in df.groupby("Country")
}

series_by_country = {}
for country, cdf in country_dfs.items():
    ts = TimeSeries.from_dataframe(
        cdf,
        time_col="Year",
        value_cols="Stroke_Deaths"
    )
    series_by_country[country] = ts

print(f"\n✓ {len(series_by_country)} time series created")

# Load model
print("\nLoading N-BEATS model...")
global_model = NBEATSModel.load("nbeats_global_model.pth")
print("✓ Model loaded!")

# Generate predictions untuk semua negara (5 tahun)
predictions_dict = {}
years_to_predict = 5

print(f"\nGenerating predictions for {len(series_by_country)} countries...")
for i, (country, ts) in enumerate(series_by_country.items(), 1):
    try:
        # Prediksi
        fc = global_model.predict(n=years_to_predict, series=ts)
        
        # Convert ke dict
        def ts_to_df(ts):
            df_temp = ts.to_dataframe().reset_index()
            cols = list(df_temp.columns)
            cols[0] = "Date"
            df_temp.columns = cols
            return df_temp
        
        hist_df = ts_to_df(ts)
        fc_df = ts_to_df(fc)
        
        hist_df.columns = ["Year", "Stroke_Deaths_Actual"]
        fc_df.columns = ["Year", "Stroke_Deaths_Forecast"]
        
        fc_df['Year'] = fc_df['Year'].dt.year
        hist_df['Year'] = hist_df['Year'].dt.year
        
        predictions_dict[country] = {
            'historical': hist_df.to_dict('records'),
            'forecast': fc_df.to_dict('records'),
            'last_historical_year': int(hist_df['Year'].max()),
            'last_historical_value': float(hist_df['Stroke_Deaths_Actual'].iloc[-1]),
            'forecast_years': f"{fc_df['Year'].min()} - {fc_df['Year'].max()}",
            'last_forecast_value': float(fc_df['Stroke_Deaths_Forecast'].iloc[-1]),
            'avg_forecast': float(fc_df['Stroke_Deaths_Forecast'].mean())
        }
        
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(series_by_country)} countries...")
    
    except Exception as e:
        print(f"  ⚠ Error for {country}: {str(e)}")
        continue

print(f"\n✓ Generated predictions for {len(predictions_dict)} countries")

# Simpan ke pickle file
with open('nbeats_predictions_cache.pkl', 'wb') as f:
    pickle.dump(predictions_dict, f)

print("✓ Predictions saved to: nbeats_predictions_cache.pkl")
print(f"✓ File size: {len(pickle.dumps(predictions_dict)) / 1024:.2f} KB")

print("\n" + "="*60)
print("DONE! Now predictions will load instantly in the web app")
print("="*60)
