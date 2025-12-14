from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
import warnings
import plotly.express as px
import plotly.graph_objects as go
import torch
import joblib
import numpy as np
from io import BytesIO
import base64

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Konfigurasi untuk menghindari warning matplotlib
plt.switch_backend('Agg')

# Path untuk folder static
STATIC_FOLDER = 'static'
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)


@app.route('/')
def index():
    """Halaman utama dengan informasi navigasi dan fitur"""
    # Baca data untuk statistik
    try:
        # Baca data timeseries untuk statistik
        df = pd.read_csv('timeseries.csv')
        df = df.rename(columns={
            'Entity': 'Country',
            'Total deaths from stroke among both sexes': 'Stroke_Deaths'
        })
        if 'Code' in df.columns:
            df = df.drop(columns=['Code'])
        
        REMOVE = ["World", "Asia", "Africa", "Europe", "European Union",
                  "North America", "South America", "Latin America and Caribbean", "Oceania"]
        df = df[~df['Country'].isin(REMOVE)]
        
        num_countries = len(df['Country'].unique())
        num_years_historical = len(df['Year'].unique())
        
        # Coba baca cluster data jika ada
        try:
            cluster_data = pd.read_csv('hasil_cluster_kmeans.csv')
            num_clusters = len(cluster_data['Cluster'].unique())
        except:
            num_clusters = 5  # default
        
        return render_template('index.html',
                             num_countries=num_countries,
                             num_clusters=num_clusters,
                             num_years_historical=num_years_historical,
                             num_years_prediction=20)  # maksimal prediksi
    except Exception as e:
        return f'<h1>Error</h1><p>Terjadi kesalahan: {str(e)}</p>'


@app.route('/kmeans')
def kmeans_visualization():
    """Route untuk menampilkan visualisasi K-Means dengan peta dunia"""
    try:
        # Membaca file data
        cluster_data = pd.read_csv('hasil_cluster_kmeans.csv')
        
        # Mapping negara untuk Plotly (beberapa nama perlu disesuaikan)
        country_mapping = {
            'United States': 'United States of America',
            'Democratic Republic of Congo': 'Dem. Rep. Congo',
            'Congo': 'Congo',
            'Tanzania': 'Tanzania',
            'Cote d\'Ivoire': 'Ivory Coast',
            'Guinea Bissau': 'Guinea-Bissau',
            'Timor': 'Timor-Leste',
            'Czech Republic': 'Czechia',
            'Macedonia': 'North Macedonia',
            'Serbia': 'Serbia',
            'South Sudan': 'S. Sudan'
        }
        
        # Replace nama negara yang perlu disesuaikan
        cluster_data['Country_Mapped'] = cluster_data['Country'].replace(country_mapping)
        
        # Filter data yang valid (bukan region/continent)
        regions_to_exclude = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 
                             'Oceania', 'World', 'Eastern Mediterranean', 'Western Pacific',
                             'South-East Asia', 'Americas', 'European Region']
        cluster_data_filtered = cluster_data[~cluster_data['Country'].isin(regions_to_exclude)]
        
        # Membuat color palette untuk cluster
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Assign colors to clusters
        cluster_data_filtered['Color'] = cluster_data_filtered['Cluster'].apply(
            lambda x: colors[x % len(colors)]
        )
        
        # Membuat peta choropleth dengan Plotly
        fig = go.Figure(data=go.Choropleth(
            locations=cluster_data_filtered['Country_Mapped'],
            locationmode='country names',
            z=cluster_data_filtered['Cluster'],
            text=cluster_data_filtered['Country'],
            colorscale=[
                [0.0, colors[0]],
                [0.25, colors[1]],
                [0.5, colors[2]],
                [0.75, colors[3]],
                [1.0, colors[4]]
            ],
            autocolorscale=False,
            reversescale=False,
            marker_line_color='darkgray',
            marker_line_width=0.5,
            colorbar_title='Cluster',
            hovertemplate='<b>%{text}</b><br>Cluster: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': 'K-Means Clustering - Peta Dunia Berdasarkan Cluster',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial Black'}
            },
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth',
                bgcolor='rgba(240,240,240,0.5)'
            ),
            width=1400,
            height=700,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Simpan sebagai HTML
        map_html_path = os.path.join(STATIC_FOLDER, 'kmeans_map.html')
        fig.write_html(map_html_path)
        
        # Statistik cluster
        cluster_stats = cluster_data_filtered.groupby('Cluster').size().to_dict()
        num_clusters = len(cluster_data_filtered['Cluster'].unique())
        num_countries = len(cluster_data_filtered)
        
        # Definisi karakteristik cluster berdasarkan data
        cluster_info = {
            0: {
                'name': 'Cluster 0 - Negara dengan Populasi Sedang',
                'description': 'Mayoritas negara di dunia dengan jumlah kematian stroke dalam kategori rendah hingga menengah',
                'examples': ['Afghanistan', 'Albania', 'Algeria', 'Australia', 'Austria'],
                'characteristics': [
                    'Jumlah kematian stroke relatif rendah hingga menengah',
                    'Mencakup mayoritas negara di seluruh benua',
                    'Populasi dan infrastruktur kesehatan bervariasi',
                    'Trend kematian stroke relatif stabil'
                ]
            },
            1: {
                'name': 'Cluster 1 - Negara dengan Populasi Sangat Tinggi',
                'description': 'Negara dengan jumlah kematian stroke tertinggi karena populasi sangat besar',
                'examples': ['China'],
                'characteristics': [
                    'Populasi sangat besar (> 1 miliar)',
                    'Jumlah kematian stroke absolut sangat tinggi',
                    'Negara dengan ekonomi besar dan berkembang pesat',
                    'Memerlukan perhatian khusus dalam pencegahan stroke'
                ]
            },
            2: {
                'name': 'Cluster 2 - Negara dengan Populasi Tinggi',
                'description': 'Negara-negara berpopulasi besar dengan jumlah kematian stroke tinggi',
                'examples': ['India', 'Indonesia', 'Russia'],
                'characteristics': [
                    'Populasi besar (ratusan juta)',
                    'Jumlah kematian stroke tinggi',
                    'Negara berkembang dengan pertumbuhan ekonomi',
                    'Tantangan dalam sistem kesehatan dan pencegahan'
                ]
            }
        }
        
        return render_template(
            'kmeans.html',
            map_file='kmeans_map.html',
            num_clusters=num_clusters,
            num_countries=num_countries,
            cluster_stats=cluster_stats,
            cluster_info=cluster_info
        )
        
    except Exception as e:
        return f'<h1>Error</h1><p>Terjadi kesalahan: {str(e)}</p><a href="/">Kembali</a>'


@app.route('/arima', methods=['GET', 'POST'])
def arima_visualization():
    """Route untuk menampilkan visualisasi ARIMA per negara dengan input custom"""
    from flask import request
    from pmdarima import auto_arima
    
    try:
        # Membaca file data historis
        df = pd.read_csv('timeseries.csv')
        
        # Rename kolom agar lebih mudah digunakan
        df = df.rename(columns={
            'Entity': 'Country',
            'Total deaths from stroke among both sexes': 'Stroke_Deaths'
        })
        
        # Hapus kolom Code jika ada
        if 'Code' in df.columns:
            df = df.drop(columns=['Code'])
        
        # Hapus baris entitas agregat (bukan negara)
        REMOVE = [
            "World", "Asia", "Africa", "Europe", "European Union",
            "North America", "South America", "Latin America and Caribbean",
            "Oceania"
        ]
        df = df[~df['Country'].isin(REMOVE)]
        
        # Preprocessing
        df = df.drop_duplicates()
        df['Stroke_Deaths'] = df.groupby('Country')['Stroke_Deaths'].transform(lambda x: x.interpolate(method='linear'))
        df['Stroke_Deaths'] = df.groupby('Country')['Stroke_Deaths'].transform(lambda x: x.fillna(x.median()))
        df['Year'] = df['Year'].astype(int)
        df['Stroke_Deaths'] = df['Stroke_Deaths'].astype(float)
        df = df.sort_values(by=['Country', 'Year']).reset_index(drop=True)
        
        # Dapatkan daftar negara
        list_negara = sorted(df['Country'].unique())
        
        # Default: gunakan parameter dari request
        if request.method == 'POST':
            selected_country = request.form.get('country', list_negara[0])
            try:
                years_to_predict = int(request.form.get('years', 5))
                if years_to_predict < 1:
                    years_to_predict = 5
                elif years_to_predict > 20:
                    years_to_predict = 20
            except:
                years_to_predict = 5
        else:
            selected_country = request.args.get('country', list_negara[0])
            years_to_predict = int(request.args.get('years', 5))
        
        # Filter data untuk negara yang dipilih
        ts = df[df['Country'] == selected_country].groupby('Year')['Stroke_Deaths'].sum()
        ts.index = pd.to_datetime(ts.index, format='%Y')
        
        # Auto ARIMA
        model = auto_arima(
            ts,
            seasonal=False,
            suppress_warnings=True,
            stepwise=True,
            error_action='ignore'
        )
        
        # Ambil p,d,q
        p, d, q = model.order
        arima_order = f"ARIMA({p}, {d}, {q})"
        
        # Prediksi n tahun ke depan
        forecast_values = model.predict(n_periods=years_to_predict)
        last_year = ts.index.year.max()
        
        # Buat DataFrame untuk hasil prediksi
        prediction_years = list(range(last_year + 1, last_year + years_to_predict + 1))
        prediction_data = pd.DataFrame({
            'Year': prediction_years,
            'Predicted_Stroke_Deaths': forecast_values
        })
        
        # Membuat visualisasi
        plt.figure(figsize=(14, 8))
        
        # Plot data historis
        plt.plot(
            ts.index.year,
            ts.values,
            label='Data Historis',
            color='#1f77b4',
            linewidth=2.5,
            marker='o',
            markersize=6
        )
        
        # Plot hasil prediksi
        plt.plot(
            prediction_data['Year'],
            prediction_data['Predicted_Stroke_Deaths'],
            label=f'Prediksi {arima_order}',
            color='#ff7f0e',
            linewidth=2.5,
            linestyle='--',
            marker='s',
            markersize=6
        )
        
        # Menandai titik transisi antara data historis dan prediksi
        plt.axvline(
            x=last_year,
            color='red',
            linestyle=':',
            linewidth=2,
            alpha=0.7,
            label='Batas Historis-Prediksi'
        )
        
        # Konfigurasi plot
        plt.xlabel('Tahun', fontsize=13, fontweight='bold')
        plt.ylabel('Jumlah Kematian Stroke', fontsize=13, fontweight='bold')
        plt.title(f'Visualisasi Data Historis dan Prediksi ARIMA\nKematian Stroke - {selected_country}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='best', fontsize=11, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Format y-axis untuk menampilkan angka dengan pemisah ribuan
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Menyimpan gambar
        image_path = os.path.join(STATIC_FOLDER, 'arima_plot.png')
        plt.savefig(image_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Informasi tambahan
        num_historical = len(ts)
        num_predictions = len(prediction_data)
        last_historical_value = ts.values[-1]
        last_prediction_value = prediction_data['Predicted_Stroke_Deaths'].iloc[-1]
        last_historical_year = last_year
        first_prediction_year = prediction_data['Year'].iloc[0]
        last_prediction_year = prediction_data['Year'].iloc[-1]
        
        # Hitung MAPE in-sample untuk evaluasi
        pred_in_sample = model.predict_in_sample()
        from sklearn.metrics import mean_absolute_percentage_error
        mape = mean_absolute_percentage_error(ts.values, pred_in_sample) * 100
        
        return render_template(
            'arima.html',
            image_file='arima_plot.png',
            num_historical=num_historical,
            num_predictions=num_predictions,
            last_historical_value=f'{last_historical_value:,.0f}',
            last_prediction_value=f'{last_prediction_value:,.0f}',
            last_historical_year=int(last_historical_year),
            first_prediction_year=int(first_prediction_year),
            last_prediction_year=int(last_prediction_year),
            years_to_predict=years_to_predict,
            selected_country=selected_country,
            countries=list_negara,
            arima_order=arima_order,
            mape=f'{mape:.2f}'
        )
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f'<h1>Error</h1><p>Terjadi kesalahan: {str(e)}</p><pre>{error_detail}</pre><a href="/">Kembali</a>'


@app.route('/dl-clustering')
def dl_clustering():
    """Route untuk menampilkan visualisasi LSTM Autoencoder Clustering"""
    try:
        # Baca hasil clustering dari LSTM
        cluster_data = pd.read_csv('hasil_clustering_lstm.csv')
        
        # Rename kolom jika perlu
        if 'Entity' in cluster_data.columns:
            cluster_data = cluster_data.rename(columns={'Entity': 'Country', 'cluster': 'Cluster'})
        
        # Mapping negara untuk Plotly
        country_mapping = {
            'United States': 'United States of America',
            'Democratic Republic of Congo': 'Dem. Rep. Congo',
            'Congo': 'Congo',
            'Tanzania': 'Tanzania',
            'Cote d\'Ivoire': 'Ivory Coast',
            'Guinea Bissau': 'Guinea-Bissau',
            'Timor': 'Timor-Leste',
            'Czech Republic': 'Czechia',
            'Macedonia': 'North Macedonia',
            'Serbia': 'Serbia',
            'South Sudan': 'S. Sudan'
        }
        
        cluster_data['Country_Mapped'] = cluster_data['Country'].replace(country_mapping)
        
        # Statistik cluster
        cluster_0_count = len(cluster_data[cluster_data['Cluster'] == 0])
        cluster_1_count = len(cluster_data[cluster_data['Cluster'] == 1])
        cluster_2_count = len(cluster_data[cluster_data['Cluster'] == 2])
        
        # Membuat peta choropleth
        colors = ['#3498db', '#e74c3c', '#f39c12']
        
        fig = go.Figure(data=go.Choropleth(
            locations=cluster_data['Country_Mapped'],
            locationmode='country names',
            z=cluster_data['Cluster'],
            text=cluster_data['Country'],
            colorscale=[
                [0.0, colors[0]],
                [0.5, colors[1]],
                [1.0, colors[2]]
            ],
            autocolorscale=False,
            reversescale=False,
            marker_line_color='darkgray',
            marker_line_width=0.5,
            colorbar_title='Cluster',
            hovertemplate='<b>%{text}</b><br>Cluster: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': 'LSTM Autoencoder Clustering - Deep Learning',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial Black'}
            },
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth',
                bgcolor='rgba(240,240,240,0.5)'
            ),
            width=1400,
            height=700,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Simpan peta
        map_html_path = os.path.join(STATIC_FOLDER, 'lstm_cluster_map.html')
        fig.write_html(map_html_path)
        
        return render_template(
            'dl_clustering.html',
            cluster_0_count=cluster_0_count,
            cluster_1_count=cluster_1_count,
            cluster_2_count=cluster_2_count
        )
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f'<h1>Error</h1><p>Terjadi kesalahan: {str(e)}</p><pre>{error_detail}</pre><a href="/">Kembali</a>'


@app.route('/dl-prediction', methods=['GET', 'POST'])
def dl_prediction():
    """Route untuk menampilkan prediksi N-BEATS (Cache-Based)"""
    from flask import request
    
    try:
        # Baca data untuk dropdown
        df = pd.read_csv('datafix.csv')
        list_negara = sorted(df['Country'].unique())
        
        # Default parameters
        years_to_predict = 5  # Fixed to 5 years
        if request.method == 'POST':
            selected_country = request.form.get('country', list_negara[0])
        else:
            selected_country = request.args.get('country', list_negara[0])
        
        prediction_data = None
        plot_data = None
        
        if request.method == 'POST':
            # Load pre-computed predictions from cache
            try:
                import pickle
                with open('nbeats_predictions_cache.pkl', 'rb') as f:
                    predictions_cache = pickle.load(f)
                
                # Load evaluation metrics
                eval_df = pd.read_csv('hasil_evaluasi_nbeats.csv')
                
                # Get predictions for selected country
                if selected_country not in predictions_cache:
                    raise ValueError(f"No predictions available for {selected_country}")
                
                country_data = predictions_cache[selected_country]
                
                # Get evaluation metrics for selected country
                country_eval = eval_df[eval_df['Country'] == selected_country]
                if not country_eval.empty:
                    mape_value = country_eval['MAPE'].values[0]
                    mae_value = country_eval['MAE'].values[0]
                    rmse_value = country_eval['RMSE'].values[0]
                else:
                    # Use global average if country not found
                    mape_value = eval_df['MAPE'].mean()
                    mae_value = eval_df['MAE'].mean()
                    rmse_value = eval_df['RMSE'].mean()
                
                # Extract historical data from cache
                hist_data = country_data['historical']
                hist_years = [item['Year'] for item in hist_data]
                hist_values = [item['Stroke_Deaths_Actual'] for item in hist_data]
                
                # Extract forecast data from cache
                forecast_data = country_data['forecast']
                
                # Limit to requested years
                forecast_data = forecast_data[:years_to_predict]
                pred_years = [item['Year'] for item in forecast_data]
                pred_values = [item['Stroke_Deaths_Forecast'] for item in forecast_data]
                
                # Siapkan data untuk template
                prediction_data = []
                for item in forecast_data:
                    prediction_data.append({
                        'Year': int(item['Year']),
                        'Predicted_Deaths': float(item['Stroke_Deaths_Forecast'])
                    })
                
                # Informasi statistik
                last_year = hist_years[-1]
                last_value = hist_values[-1]
                prediction_years = f"{pred_years[0]} - {pred_years[-1]}"
                last_prediction = pred_values[-1]
                avg_prediction = sum(pred_values) / len(pred_values)
                
                # Hitung trend
                if last_prediction > last_value:
                    trend = f"+{((last_prediction / last_value - 1) * 100):.1f}%"
                else:
                    trend = f"{((last_prediction / last_value - 1) * 100):.1f}%"
                
                # Buat plot
                plt.figure(figsize=(14, 8))
                
                plt.plot(
                    hist_years,
                    hist_values,
                    marker="o",
                    label="Historical",
                    linewidth=2.5,
                    color='#1f77b4'
                )
                
                plt.plot(
                    pred_years,
                    pred_values,
                    marker="o",
                    linestyle="--",
                    label=f"Forecast ({years_to_predict} years)",
                    linewidth=2.5,
                    color='#ff7f0e'
                )
                
                plt.axvline(x=last_year, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Batas Historis-Prediksi')
                
                plt.title(f"Stroke Deaths per Year — {selected_country}", fontsize=16, fontweight='bold')
                plt.xlabel("Year", fontsize=13, fontweight='bold')
                plt.ylabel("Stroke Deaths", fontsize=13, fontweight='bold')
                plt.legend(loc='best', fontsize=11)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Format y-axis
                ax = plt.gca()
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
                
                # Convert plot to base64
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                return render_template(
                    'dl_prediction.html',
                    countries=list_negara,
                    selected_country=selected_country,
                    years_to_predict=years_to_predict,
                    prediction_data=prediction_data,
                    last_year=last_year,
                    last_value=last_value,
                    prediction_years=prediction_years,
                    last_prediction=last_prediction,
                    avg_prediction=avg_prediction,
                    trend=trend,
                    plot_data=plot_data,
                    mape_value=mape_value,
                    mae_value=mae_value,
                    rmse_value=rmse_value
                )
            
            except Exception as model_error:
                # Jika error saat load predictions, tampilkan error message
                import traceback
                error_detail = traceback.format_exc()
                print(f"ERROR: {error_detail}")
                
                return render_template(
                    'dl_prediction.html',
                    countries=list_negara,
                    selected_country=selected_country,
                    years_to_predict=years_to_predict,
                    error_message=f"Prediksi gagal dimuat. Error: {str(model_error)}"
                )
        
        return render_template(
            'dl_prediction.html',
            countries=list_negara,
            selected_country=list_negara[0],
            years_to_predict=5
        )
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f'<h1>Error</h1><p>Terjadi kesalahan: {str(e)}</p><pre>{error_detail}</pre><a href="/">Kembali</a>'


@app.route('/hybrid-prediction', methods=['GET', 'POST'])
def hybrid_prediction():
    """Route untuk Hybrid Clustering per Tahun (Tanpa ARIMA - Langsung dari Data)"""
    from flask import request
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import geopandas as gpd
    
    try:
        # Load data pivot (negara x tahun)
        df_pivot = pd.read_csv('pivot_kmeans.csv')
        
        # Default year atau dari form
        if request.method == 'POST':
            selected_year = int(request.form.get('year', 2022))
        else:
            selected_year = 2022  # Default tahun pertama kali load
        
        map_file = None
        cluster_counts = None
        error_message = None
        
        try:
            # Step 1: Prediksi menggunakan ARIMA untuk semua negara
            print(f"Generating predictions for year {selected_year}...")
            predictions = {}
            failed_countries = []
            
            for country in df['Entity'].unique():
                country_data = df[df['Entity'] == country].sort_values('Year')
                
                if len(country_data) < 3:
                    failed_countries.append(f"{country} (data < 3)")
                    continue
                
                try:
                    # Predict ke tahun yang dipilih
                    last_year = int(country_data['Year'].max())
                    n_periods = selected_year - last_year
                    
                    if n_periods > 0:
                        # Fit ARIMA model
                        model = auto_arima(
                            country_data['Total deaths from stroke among both sexes'],
                            seasonal=False,
                            stepwise=True,
                            suppress_warnings=True,
                            error_action='ignore',
                            max_p=3, max_q=3, max_d=2,
                            trace=False
                        )
                        forecast = model.predict(n_periods=n_periods)
                        predictions[country] = float(forecast[-1])
                    else:
                        # Jika tahun sudah ada di data, gunakan nilai aktual
                        year_data = country_data[country_data['Year'] == selected_year]
                        if len(year_data) > 0:
                            predictions[country] = float(year_data['Total deaths from stroke among both sexes'].values[0])
                        else:
                            # Gunakan nilai terakhir jika tahun tidak tersedia
                            predictions[country] = float(country_data['Total deaths from stroke among both sexes'].iloc[-1])
                except Exception as e:
                    failed_countries.append(f"{country} ({str(e)[:50]})")
                    continue
            
            print(f"✓ Successfully predicted {len(predictions)} countries")
            if failed_countries:
                print(f"⚠ Failed for {len(failed_countries)} countries")
            
            # Step 2: Clustering berdasarkan nilai prediksi
            pred_df = pd.DataFrame(list(predictions.items()), columns=['Country', 'Predicted_Deaths'])
            
            # Check if we have enough predictions
            if len(pred_df) < 3:
                raise ValueError(f"Tidak cukup data prediksi. Hanya {len(pred_df)} negara yang berhasil diprediksi. Minimum 3 negara diperlukan untuk clustering.")
            
            # Standardize predictions
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(pred_df[['Predicted_Deaths']])
            
            # K-Means clustering (3 clusters)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            pred_df['Cluster'] = kmeans.fit_predict(X_scaled)
            
            # Step 3: Buat peta dengan Plotly
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            
            # Merge dengan data prediksi
            world = world.merge(pred_df, left_on='name', right_on='Country', how='left')
            world['Cluster'] = world['Cluster'].fillna(-1).astype(int)
            
            # Buat color mapping
            color_map = {
                0: '#00b894',  # Hijau - Rendah
                1: '#fdcb6e',  # Kuning - Sedang
                2: '#d63031',  # Merah - Tinggi
                -1: '#dfe6e9'  # Abu-abu - No data
            }
            
            world['Color'] = world['Cluster'].map(color_map)
            
            # Create Plotly figure
            fig = go.Figure()
            
            for cluster in [0, 1, 2, -1]:
                cluster_data = world[world['Cluster'] == cluster]
                
                if cluster == -1:
                    name = 'No Data'
                elif cluster == 0:
                    name = 'Cluster 0 (Rendah)'
                elif cluster == 1:
                    name = 'Cluster 1 (Sedang)'
                else:
                    name = 'Cluster 2 (Tinggi)'
                
                fig.add_trace(go.Choropleth(
                    locations=cluster_data['iso_a3'],
                    z=cluster_data['Cluster'],
                    text=cluster_data['name'],
                    customdata=cluster_data['Predicted_Deaths'],
                    hovertemplate='<b>%{text}</b><br>Cluster: ' + name + '<br>Predicted Deaths: %{customdata:,.0f}<extra></extra>',
                    colorscale=[[0, color_map[cluster]], [1, color_map[cluster]]],
                    showscale=False,
                    name=name
                ))
            
            fig.update_layout(
                title={
                    'text': f'Hybrid Clustering & Prediction - Tahun {selected_year}',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 24, 'color': '#2d3436', 'family': 'Arial Black'}
                },
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    projection_type='natural earth',
                    bgcolor='rgba(255,255,255,0)'
                ),
                height=600,
                margin=dict(l=0, r=0, t=80, b=0),
                paper_bgcolor='#f8f9fa',
                plot_bgcolor='#f8f9fa'
            )
            
            # Save map
            map_filename = f'hybrid_map_{selected_year}.html'
            map_path = os.path.join(STATIC_FOLDER, map_filename)
            fig.write_html(map_path)
            
            map_file = map_filename
            
            # Count clusters
            cluster_counts = pred_df['Cluster'].value_counts().to_dict()
            
            print(f"✓ Hybrid map generated for year {selected_year}")
            
        except Exception as e:
            import traceback
            error_message = f"Error generating hybrid prediction: {str(e)}"
            print(f"ERROR: {traceback.format_exc()}")
        
        return render_template(
            'hybrid_prediction.html',
            selected_year=selected_year,
            map_file=map_file,
            cluster_counts=cluster_counts,
            error_message=error_message
        )
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f'<h1>Error</h1><p>Terjadi kesalahan: {str(e)}</p><pre>{error_detail}</pre><a href="/">Kembali</a>'


if __name__ == '__main__':
    # Menjalankan aplikasi Flask
    print("="*60)
    print("Aplikasi Flask - Machine Learning & Deep Learning")
    print("="*60)
    print("Akses aplikasi di:")
    print("- Dashboard: http://127.0.0.1:5000/")
    print("\nMachine Learning:")
    print("  - K-Means: http://127.0.0.1:5000/kmeans")
    print("  - ARIMA: http://127.0.0.1:5000/arima")
    print("  - Hybrid: http://127.0.0.1:5000/hybrid-prediction")
    print("\nDeep Learning:")
    print("  - LSTM Clustering: http://127.0.0.1:5000/dl-clustering")
    print("  - N-BEATS Prediction: http://127.0.0.1:5000/dl-prediction")
    print("="*60)
    app.run(debug=True, host='127.0.0.1', port=5000)
