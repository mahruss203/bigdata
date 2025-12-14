import pandas as pd
import plotly.graph_objects as go

# Baca data clustering LSTM
cluster_data = pd.read_csv('hasil_clustering_lstm.csv')

# Rename kolom
cluster_data = cluster_data.rename(columns={'Entity': 'Country', 'cluster': 'Cluster'})

# Mapping negara
country_mapping = {
    'United States': 'United States of America',
    'Democratic Republic of Congo': 'Dem. Rep. Congo',
    'Congo': 'Congo',
    'Tanzania': 'Tanzania',
    "Cote d'Ivoire": 'Ivory Coast',
    'Guinea Bissau': 'Guinea-Bissau',
    'Timor': 'Timor-Leste',
    'Czech Republic': 'Czechia',
    'Macedonia': 'North Macedonia',
    'Serbia': 'Serbia',
    'South Sudan': 'S. Sudan'
}

cluster_data['Country_Mapped'] = cluster_data['Country'].replace(country_mapping)

# Warna cluster
colors = ['#3498db', '#e74c3c', '#f39c12']

# Buat peta
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

# Simpan
fig.write_html('static/lstm_cluster_map.html')
print('✓ Peta LSTM clustering berhasil dibuat: static/lstm_cluster_map.html')
