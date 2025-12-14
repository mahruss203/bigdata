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
    """Main page with navigation information and features"""
    # Read data for statistics
    try:
        # Read timeseries data for statistics
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
        
        # Try to read cluster data if available
        try:
            cluster_data = pd.read_csv('hasil_cluster_kmeans.csv')
            num_clusters = len(cluster_data['Cluster'].unique())
        except:
            num_clusters = 5  # default
        
        return render_template('index.html',
                             num_countries=num_countries,
                             num_clusters=num_clusters,
                             num_years_historical=num_years_historical,
                             num_years_prediction=20)  # maximum prediction
    except Exception as e:
        return f'<h1>Error</h1><p>An error occurred: {str(e)}</p>'


@app.route('/kmeans')
def kmeans_visualization():
    """Route to display K-Means visualization with world map"""
    try:
        # Read data files
        cluster_data = pd.read_csv('hasil_cluster_kmeans.csv')
        
        # Country mapping for Plotly (some names need to be adjusted)
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
                'text': 'K-Means Clustering - World Map Based on Clusters',
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
        
        # Save as HTML
        map_html_path = os.path.join(STATIC_FOLDER, 'kmeans_map.html')
        fig.write_html(map_html_path)
        
        # Cluster statistics
        cluster_stats = cluster_data_filtered.groupby('Cluster').size().to_dict()
        num_clusters = len(cluster_data_filtered['Cluster'].unique())
        num_countries = len(cluster_data_filtered)
        
        # Cluster characteristics definition based on data
        cluster_info = {
            0: {
                'name': 'Cluster 0 - Countries with Medium Population',
                'description': 'Majority of countries in the world with low to medium stroke death counts',
                'examples': ['Afghanistan', 'Albania', 'Algeria', 'Australia', 'Austria'],
                'characteristics': [
                    'Relatively low to medium stroke death counts',
                    'Includes majority of countries across all continents',
                    'Varied population and healthcare infrastructure',
                    'Relatively stable stroke death trends'
                ]
            },
            1: {
                'name': 'Cluster 1 - Countries with Very High Population',
                'description': 'Countries with highest stroke deaths due to very large population',
                'examples': ['China'],
                'characteristics': [
                    'Very large population (> 1 billion)',
                    'Very high absolute stroke death counts',
                    'Countries with large and rapidly developing economies',
                    'Requires special attention in stroke prevention'
                ]
            },
            2: {
                'name': 'Cluster 2 - Countries with High Population',
                'description': 'Countries with large populations with high stroke death counts',
                'examples': ['India', 'Indonesia', 'Russia'],
                'characteristics': [
                    'Large population (hundreds of millions)',
                    'High stroke death counts',
                    'Developing countries with economic growth',
                    'Challenges in healthcare systems and prevention'
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
        return f'<h1>Error</h1><p>An error occurred: {str(e)}</p><a href="/">Back</a>'


@app.route('/arima', methods=['GET', 'POST'])
def arima_visualization():
    """Route to display ARIMA visualization per country with custom input"""
    from flask import request
    from pmdarima import auto_arima
    
    try:
        # Read historical data file
        df = pd.read_csv('timeseries.csv')
        
        # Rename columns for easier use
        df = df.rename(columns={
            'Entity': 'Country',
            'Total deaths from stroke among both sexes': 'Stroke_Deaths'
        })
        
        # Remove Code column if exists
        if 'Code' in df.columns:
            df = df.drop(columns=['Code'])
        
        # Remove aggregated entity rows (not countries)
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
        
        # Get list of countries
        list_negara = sorted(df['Country'].unique())
        
        # Default: use parameters from request
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
        
        # Filter data for selected country
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
        
        # Get p,d,q
        p, d, q = model.order
        arima_order = f"ARIMA({p}, {d}, {q})"
        
        # Predict n years ahead
        forecast_values = model.predict(n_periods=years_to_predict)
        last_year = ts.index.year.max()
        
        # Create DataFrame for prediction results
        prediction_years = list(range(last_year + 1, last_year + years_to_predict + 1))
        prediction_data = pd.DataFrame({
            'Year': prediction_years,
            'Predicted_Stroke_Deaths': forecast_values
        })
        
        # Create visualization
        plt.figure(figsize=(14, 8))
        
        # Plot historical data
        plt.plot(
            ts.index.year,
            ts.values,
            label='Historical Data',
            color='#1f77b4',
            linewidth=2.5,
            marker='o',
            markersize=6
        )
        
        # Plot prediction results
        plt.plot(
            prediction_data['Year'],
            prediction_data['Predicted_Stroke_Deaths'],
            label=f'Prediction {arima_order}',
            color='#ff7f0e',
            linewidth=2.5,
            linestyle='--',
            marker='s',
            markersize=6
        )
        
        # Mark transition point between historical data and prediction
        plt.axvline(
            x=last_year,
            color='red',
            linestyle=':',
            linewidth=2,
            alpha=0.7,
            label='Historical-Prediction Boundary'
        )
        
        # Plot configuration
        plt.xlabel('Year', fontsize=13, fontweight='bold')
        plt.ylabel('Stroke Deaths Count', fontsize=13, fontweight='bold')
        plt.title(f'Historical Data and ARIMA Prediction Visualization\nStroke Deaths - {selected_country}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='best', fontsize=11, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Format y-axis to display numbers with thousand separators
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Save image
        image_path = os.path.join(STATIC_FOLDER, 'arima_plot.png')
        plt.savefig(image_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Additional information
        num_historical = len(ts)
        num_predictions = len(prediction_data)
        last_historical_value = ts.values[-1]
        last_prediction_value = prediction_data['Predicted_Stroke_Deaths'].iloc[-1]
        last_historical_year = last_year
        first_prediction_year = prediction_data['Year'].iloc[0]
        last_prediction_year = prediction_data['Year'].iloc[-1]
        
        # Calculate MAPE in-sample for evaluation
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
        return f'<h1>Error</h1><p>An error occurred: {str(e)}</p><pre>{error_detail}</pre><a href="/">Back</a>'


@app.route('/arima-kmeans', methods=['GET', 'POST'])
def arima_kmeans():
    """Route for ARIMA + K-Means Hybrid Clustering"""
    from flask import request
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import pickle
    
    try:
        # Read data for dropdown
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
        
        list_negara = sorted(df['Country'].unique())
        
        # Default parameters
        if request.method == 'POST':
            try:
                years_to_predict = int(request.form.get('years', 3))
                if years_to_predict < 1:
                    years_to_predict = 1
                elif years_to_predict > 5:
                    years_to_predict = 5
            except:
                years_to_predict = 3
        else:
            years_to_predict = 3
        
        prediction_data = None
        cluster_stats = None
        map_html = None
        
        # Generate map untuk GET atau POST
        # Untuk GET: gunakan data historis terbaru
        # Untuk POST: gunakan prediksi ARIMA
        
        if request.method == 'GET':
            # Clustering berdasarkan data historis (tahun terakhir)
            latest_year = df['Year'].max()
            latest_data = df[df['Year'] == latest_year].copy()
            
            # Aggregate per negara
            cluster_data = latest_data.groupby('Country')['Stroke_Deaths'].mean().reset_index()
            cluster_data.columns = ['Country', 'Value']
            base_year = latest_year
            
        else:  # POST request
            # Load ARIMA cache
            try:
                with open('arima_models_cache.pkl', 'rb') as f:
                    arima_cache = pickle.load(f)
            except FileNotFoundError:
                return render_template(
                    'arima_kmeans.html',
                    years_to_predict=years_to_predict,
                    error_message="ARIMA cache not created yet. Run generate_arima_cache.py first."
                )
            
            # Generate predictions for all countries
            predictions_dict = {}
            base_year = None
            
            for country in list_negara:
                if country in arima_cache:
                    model_data = arima_cache[country]
                    model = model_data['model']
                    last_year = model_data['last_year']
                    if base_year is None:
                        base_year = last_year
                    
                    # Prediction
                    forecast = model.predict(n_periods=years_to_predict)
                    avg_prediction = forecast.mean()
                    
                    predictions_dict[country] = {
                        'avg_prediction': avg_prediction,
                        'last_prediction': forecast[-1],
                        'predictions': forecast.tolist()
                    }
            
            # Create DataFrame for clustering
            cluster_data = pd.DataFrame([
                {'Country': country, 'Value': data['avg_prediction']}
                for country, data in predictions_dict.items()
            ])
            
        # Normalize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data[['Value']])
        
        # K-Means clustering
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_data['Cluster'] = kmeans.fit_predict(scaled_data)
        
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
        
        # Cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_countries = cluster_data[cluster_data['Cluster'] == i]
            cluster_stats[i] = {
                'count': len(cluster_countries),
                'avg_prediction': cluster_countries['Value'].mean(),
                'min_prediction': cluster_countries['Value'].min(),
                'max_prediction': cluster_countries['Value'].max(),
                'countries': cluster_countries['Country'].tolist()[:10]  # Top 10
            }
        
        # Create choropleth map
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Yellow, Red
        
        # Different title for GET vs POST
        if request.method == 'GET':
            map_title = f'K-Means Clustering - Historical Data (Year {latest_year})'
            hover_label = 'Stroke Deaths'
        else:
            map_title = f'ARIMA + K-Means Clustering - {years_to_predict} Year Prediction'
            hover_label = 'Avg Prediction'
        
        fig = go.Figure(data=go.Choropleth(
            locations=cluster_data['Country_Mapped'],
            locationmode='country names',
            z=cluster_data['Cluster'],
            text=cluster_data['Country'],
            customdata=cluster_data['Value'],
            colorscale=[
                [0.0, colors[0]],
                [0.5, colors[1]],
                [1.0, colors[2]]
            ],
            autocolorscale=False,
            reversescale=False,
            marker_line_color='#2c3e50',
            marker_line_width=1.5,
            colorbar_title='Cluster',
            hovertemplate=f'<b>%{{text}}</b><br>Cluster: %{{z}}<br>{hover_label}: %{{customdata:,.0f}}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': map_title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial Black'}
            },
            geo=dict(
                showframe=True,
                showcoastlines=True,
                coastlinecolor='#34495e',
                coastlinewidth=1,
                showcountries=True,
                countrycolor='#2c3e50',
                countrywidth=1.5,
                projection_type='natural earth',
                bgcolor='rgba(240,240,240,0.5)',
                landcolor='rgba(230,230,230,0.3)',
                oceancolor='rgba(173,216,230,0.3)'
            ),
            width=1400,
            height=700,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Simpan peta
        map_html_path = os.path.join(STATIC_FOLDER, 'arima_kmeans_map.html')
        fig.write_html(map_html_path)
        map_html = 'arima_kmeans_map.html'
        
        # Informasi tambahan
        total_countries = len(cluster_data)
        
        if request.method == 'GET':
            prediction_year = latest_year
        else:
            prediction_year = last_year + years_to_predict
        
        return render_template(
            'arima_kmeans.html',
            years_to_predict=years_to_predict,
            map_html=map_html,
            cluster_stats=cluster_stats,
            total_countries=total_countries,
            prediction_year=prediction_year,
            n_clusters=n_clusters,
            is_prediction=(request.method == 'POST'),
            base_year=base_year
        )
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f'<h1>Error</h1><p>An error occurred: {str(e)}</p><pre>{error_detail}</pre><a href="/">Back to Home</a>'


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
        return f'<h1>Error</h1><p>An error occurred: {str(e)}</p><pre>{error_detail}</pre><a href="/">Back to Home</a>'


@app.route('/dl-hybrid', methods=['GET', 'POST'])
def dl_hybrid():
    """Route untuk N-BEATS + LSTM Autoencoder Hybrid"""
    from flask import request
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import torch
    import torch.nn as nn
    
    try:
        # Baca data untuk prediksi
        df = pd.read_csv('datafix.csv')
        
        # GET request: tampilkan clustering dari data historis terbaru
        if request.method == 'GET':
            # Load hasil clustering LSTM yang sudah ada
            cluster_data = pd.read_csv('hasil_clustering_lstm.csv')
            
            if 'Entity' in cluster_data.columns:
                cluster_data = cluster_data.rename(columns={'Entity': 'Country', 'cluster': 'Cluster'})
            
            # Mapping negara
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
            cluster_stats = {}
            for i in range(3):
                cluster_stats[i] = len(cluster_data[cluster_data['Cluster'] == i])
            
            # Buat peta
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
                marker_line_width=1.5,
                colorbar_title='Cluster',
                hovertemplate='<b>%{text}</b><br>Cluster: %{z}<extra></extra>'
            ))
            
            fig.update_layout(
                title={
                    'text': 'N-BEATS & LSTM Hybrid - Data Historis (2021)',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial Black'}
                },
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    projection_type='natural earth',
                    bgcolor='rgba(240,240,240,0.5)',
                    showcountries=True,
                    countrywidth=1.5
                ),
                width=1400,
                height=700,
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            map_html_path = os.path.join(STATIC_FOLDER, 'dl_hybrid_map.html')
            fig.write_html(map_html_path)
            
            # Dapatkan tahun terakhir untuk base_year
            latest_year = df['Year'].max()
            base_year = latest_year
            
            return render_template(
                'dl_hybrid.html',
                cluster_stats=cluster_stats,
                years_to_predict=None,
                base_year=base_year
            )
        
        # POST request: generate prediksi dan re-clustering
        else:
            years_to_predict = int(request.form.get('years', 1))
            
            # Load N-BEATS predictions cache
            import pickle
            with open('nbeats_predictions_cache.pkl', 'rb') as f:
                predictions_cache = pickle.load(f)
            
            # Kumpulkan prediksi untuk semua negara
            prediction_results = []
            countries = []
            
            for country in df['Country'].unique():
                if country in predictions_cache:
                    forecast_data = predictions_cache[country]['forecast']
                    
                    # Ambil nilai prediksi sesuai tahun yang diminta
                    if years_to_predict <= len(forecast_data):
                        pred_value = forecast_data[years_to_predict - 1]['Stroke_Deaths_Forecast']
                        prediction_results.append(pred_value)
                        countries.append(country)
            
            # Buat DataFrame untuk clustering
            pred_df = pd.DataFrame({
                'Country': countries,
                'Predicted_Deaths': prediction_results
            })
            
            # Normalisasi data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(pred_df[['Predicted_Deaths']])
            
            # K-Means clustering (3 clusters)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            pred_df['Cluster'] = kmeans.fit_predict(scaled_data)
            
            # Mapping negara
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
            
            pred_df['Country_Mapped'] = pred_df['Country'].replace(country_mapping)
            
            # Statistik cluster
            cluster_stats = {}
            for i in range(3):
                cluster_stats[i] = len(pred_df[pred_df['Cluster'] == i])
            
            # Buat peta
            colors = ['#3498db', '#e74c3c', '#f39c12']
            
            fig = go.Figure(data=go.Choropleth(
                locations=pred_df['Country_Mapped'],
                locationmode='country names',
                z=pred_df['Cluster'],
                text=pred_df['Country'],
                customdata=pred_df['Predicted_Deaths'],
                colorscale=[
                    [0.0, colors[0]],
                    [0.5, colors[1]],
                    [1.0, colors[2]]
                ],
                autocolorscale=False,
                reversescale=False,
                marker_line_color='darkgray',
                marker_line_width=1.5,
                colorbar_title='Cluster',
                hovertemplate='<b>%{text}</b><br>Cluster: %{z}<br>Prediksi: %{customdata:,.0f}<extra></extra>'
            ))
            
            # Dapatkan tahun terakhir dan hitung tahun prediksi
            latest_year = df['Year'].max()
            base_year = latest_year
            prediction_year = latest_year + years_to_predict
            
            fig.update_layout(
                title={
                    'text': f'N-BEATS & LSTM Hybrid - Prediksi Tahun {prediction_year}',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial Black'}
                },
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    projection_type='natural earth',
                    bgcolor='rgba(240,240,240,0.5)',
                    showcountries=True,
                    countrywidth=1.5
                ),
                width=1400,
                height=700,
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            map_html_path = os.path.join(STATIC_FOLDER, 'dl_hybrid_map.html')
            fig.write_html(map_html_path)
            
            return render_template(
                'dl_hybrid.html',
                cluster_stats=cluster_stats,
                years_to_predict=years_to_predict,
                base_year=base_year
            )
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f'<h1>Error</h1><p>An error occurred: {str(e)}</p><pre>{error_detail}</pre><a href="/">Back to Home</a>'


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
    print("  - ARIMA + K-Means Hybrid: http://127.0.0.1:5000/arima-kmeans")
    print("\nDeep Learning:")
    print("  - LSTM Clustering: http://127.0.0.1:5000/dl-clustering")
    print("  - N-BEATS Prediction: http://127.0.0.1:5000/dl-prediction")
    print("  - N-BEATS + LSTM Hybrid: http://127.0.0.1:5000/dl-hybrid")
    print("="*60)
    
    # Get local IP address
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print("\n" + "="*60)
        print("🌐 NETWORK ACCESS")
        print("="*60)
        print(f"Local Access: http://127.0.0.1:5000")
        print(f"Network Access (WiFi): http://{local_ip}:5000")
        print("\nTo access from other devices on the same WiFi:")
        print(f"  1. Connect to the same WiFi network")
        print(f"  2. Open browser and go to: http://{local_ip}:5000")
        print("="*60)
    except:
        pass
    
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
