# Aplikasi Visualisasi Machine Learning & Deep Learning dengan Flask

Aplikasi web Flask untuk visualisasi hasil analisis K-Means Clustering, prediksi ARIMA Time Series, LSTM Autoencoder Clustering, N-BEATS Forecasting, dan **ARIMA + K-Means Hybrid Clustering**.

## 📁 Struktur Folder

```
UAS/
├── app.py                              # Aplikasi Flask utama
├── generate_arima_cache.py             # Script pre-compute ARIMA models
├── arima_models_cache.pkl              # Cache ARIMA models (generated)
├── arima_cache_metadata.pkl            # Metadata cache (generated)
├── requirements.txt                    # Dependencies Python
├── README.md                           # Dokumentasi utama
├── ARIMA_KMEANS_GUIDE.md              # Panduan fitur hybrid
├── templates/                          # Folder template HTML
│   ├── index.html                     # Dashboard utama
│   ├── kmeans.html                    # Template K-Means
│   ├── arima.html                     # Template ARIMA
│   ├── arima_kmeans.html              # Template ARIMA + K-Means Hybrid (NEW!)
│   ├── dl_clustering.html             # Template LSTM Clustering
│   └── dl_prediction.html             # Template N-BEATS
├── static/                            # Folder output visualisasi
│   ├── kmeans_map.html               # Peta K-Means
│   ├── arima_plot.png                # Plot ARIMA
│   ├── arima_kmeans_map.html         # Peta Hybrid (NEW!)
│   ├── lstm_cluster_map.html         # Peta LSTM
│   └── ...
└── Data CSV & Models:
    ├── timeseries.csv
    ├── datafix.csv
    ├── hasil_cluster_kmeans.csv
    ├── hasil_clustering_lstm.csv
    ├── nbeats_predictions_cache.pkl
    └── ...
```

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Pre-compute ARIMA Models (Hanya Sekali)

**⚠️ PENTING**: Jalankan ini sebelum menggunakan fitur ARIMA + K-Means Hybrid

```powershell
python generate_arima_cache.py
```

Proses ini akan:

- Training ARIMA model untuk 194 negara
- Memakan waktu ~3 menit
- Menghasilkan file cache untuk performa cepat

### 3. Jalankan Aplikasi Flask

```powershell
python app.py
```

### 4. Akses Aplikasi

Buka browser dan akses:

**Dashboard:**

- http://127.0.0.1:5000/

**Machine Learning:**

- http://127.0.0.1:5000/kmeans
- http://127.0.0.1:5000/arima
- http://127.0.0.1:5000/arima-kmeans ✨ **FITUR BARU!**

**Deep Learning:**

- http://127.0.0.1:5000/dl-clustering
- http://127.0.0.1:5000/dl-prediction

## 📊 Fitur Aplikasi

### 🗺️ K-Means Clustering (`/kmeans`)

- **Peta Dunia Interaktif**: Visualisasi choropleth dengan Plotly
- **Warna berdasarkan Cluster**: Setiap negara diwarnai sesuai clusternya
- **Hover Information**: Detail negara saat kursor di atas peta
- **Zoom & Pan**: Navigasi interaktif pada peta
- **Statistik per Cluster**: Jumlah negara di setiap cluster

### 📈 ARIMA Time Series (`/arima`)

- **Line Chart**: Grafik data historis dan prediksi
- **Custom Parameters**: Pilih negara dan jumlah tahun prediksi
- **Dual Plot**: Historis (biru solid) vs Prediksi (oranye dash)
- **Batas Transisi**: Garis merah vertikal menandai cut-off point
- **Auto ARIMA**: Otomatis memilih order (p,d,q) terbaik
- **MAPE Evaluation**: Metric evaluasi model

### 🔮 ARIMA + K-Means Hybrid (`/arima-kmeans`) ✨ **FITUR BARU!**

**Fitur unggulan yang menggabungkan prediksi dan clustering:**

#### Cara Kerja:

1. **User Input**: Pilih jumlah tahun prediksi (1-5 tahun) via slider interaktif
2. **Load Cache**: Sistem load pre-trained ARIMA models dari cache (cepat!)
3. **Generate Predictions**: Prediksi untuk semua 194 negara secara paralel
4. **Dynamic Re-clustering**: K-Means clustering berdasarkan nilai prediksi
5. **Interactive Map**: Peta dunia dengan warna cluster otomatis ter-update

#### Keunggulan:

- ⚡ **Super Fast**: Model sudah di-cache, tidak perlu training ulang
- 🎯 **Flexible**: Pilih 1-5 tahun prediksi sesuai kebutuhan
- 🌍 **Interactive**: Peta dunia responsif dengan hover details
- 📊 **Comprehensive Stats**: Statistik lengkap per cluster
- 🎨 **Color Coded**: Hijau (rendah), Kuning (sedang), Merah (tinggi)

#### Use Case:

- Analisis risiko stroke masa depan per negara
- Identifikasi negara high-risk untuk intervensi
- Perbandingan skenario jangka pendek vs jangka panjang
- Policy making berbasis prediksi data-driven

### 🧠 LSTM Autoencoder Clustering (`/dl-clustering`)

- **Deep Learning**: Neural network untuk ekstraksi fitur
- **Latent Space**: Representasi 10-dimensional dari time series
- **3 Clusters**: Pengelompokan berbasis pattern temporal
- **World Map**: Visualisasi geografis hasil clustering

### 📊 N-BEATS Forecasting (`/dl-prediction`)

- **Neural Forecasting**: State-of-the-art time series prediction
- **Cache-Based**: Pre-computed predictions untuk 194 negara
- **5-Year Horizon**: Prediksi sampai 5 tahun ke depan
- **MAPE/MAE/RMSE**: Multiple evaluation metrics
- **Interactive Plots**: Grafik historis vs forecast

## 🛠️ Teknologi yang Digunakan

### Backend

- **Flask**: Web framework Python
- **Pandas**: Data manipulation dan analysis
- **pmdarima**: Auto ARIMA implementation
- **scikit-learn**: K-Means, StandardScaler, PCA
- **Plotly**: Interactive visualizations
- **PyTorch**: Deep learning framework
- **Darts**: Time series forecasting library

### Frontend

- **HTML5/CSS3**: Modern responsive design
- **JavaScript**: Interactive components (slider, toggle)
- **Jinja2**: Template engine (built-in Flask)
- **Plotly.js**: Interactive charts dan maps

### Data Science

- **Matplotlib/Seaborn**: Static visualizations
- **NumPy**: Numerical computations
- **pickle**: Model serialization
- **tqdm**: Progress bars untuk batch processing

## 📝 Penjelasan Route

### `/` - Dashboard Utama

Landing page dengan navigasi ke semua fitur, toggle ML/DL, dan statistik overview.

### `/kmeans` - K-Means Clustering

1. Load data cluster dari CSV
2. Generate Plotly choropleth map
3. Mapping nama negara untuk compatibility
4. Display statistik per cluster
5. Render template dengan interactive map

### `/arima` - ARIMA Prediction

1. Load data timeseries historis
2. User pilih negara & tahun prediksi
3. Auto ARIMA training (otomatis cari order terbaik)
4. Generate forecast
5. Plot historis + prediksi dengan matplotlib
6. Display MAPE evaluation

### `/arima-kmeans` - ARIMA + K-Means Hybrid ✨

1. User pilih tahun prediksi via slider (1-5 tahun)
2. Load ARIMA models dari cache (fast!)
3. Generate predictions untuk semua negara
4. Calculate average prediction per country
5. StandardScaler normalization
6. K-Means clustering (n_clusters=3)
7. Generate interactive world map dengan Plotly
8. Display cluster statistics & country examples

**Flow Diagram:**

```
User Input (1-5 years)
    ↓
Load Cache (arima_models_cache.pkl)
    ↓
Parallel Predictions (194 countries)
    ↓
Average Calculation
    ↓
StandardScaler Normalization
    ↓
K-Means Clustering (k=3)
    ↓
Plotly Choropleth Map
    ↓
Display Results + Stats
```

### `/dl-clustering` - LSTM Autoencoder

1. Load pre-computed LSTM clustering results
2. Map country names untuk Plotly
3. Generate world map dengan 3 clusters
4. Display cluster statistics

### `/dl-prediction` - N-BEATS Forecasting

1. Load N-BEATS predictions dari cache
2. User pilih negara
3. Display forecast plot (5 years)
4. Show evaluation metrics (MAPE, MAE, RMSE)

## 🎨 Desain Interface

- **Responsive Design**: Tampilan menyesuaikan berbagai ukuran layar
- **Gradient Background**: Background modern dengan warna gradient
- **Info Cards**: Kartu informasi dengan hover effect
- **Navigation Buttons**: Tombol navigasi dengan animasi
- **Professional Layout**: Tata letak bersih dan profesional

## ⚠️ Catatan Penting

1. **ARIMA Cache**: Wajib jalankan `python generate_arima_cache.py` sebelum menggunakan fitur `/arima-kmeans`
2. **File CSV**: Pastikan semua CSV berada di root folder yang sama dengan `app.py`
3. **Folder Static**: Akan berisi HTML/PNG output yang di-generate otomatis
4. **Auto Refresh**: Visualisasi akan di-refresh setiap kali route diakses
5. **Debug Mode**: Aplikasi berjalan dalam debug mode untuk development
6. **Cache Size**: File `arima_models_cache.pkl` berukuran ~10-20 MB (normal)
7. **Browser**: Recommended Chrome/Firefox untuk kompatibilitas Plotly terbaik

## 🔧 Troubleshooting

### Error: "ARIMA cache belum dibuat"

**Solusi**: Jalankan `python generate_arima_cache.py` terlebih dahulu

### Error: File CSV tidak ditemukan

**Solusi**: Pastikan semua CSV ada di folder yang sama dengan `app.py`

### Error: Module not found

**Solusi**: Jalankan `pip install -r requirements.txt`

### Port 5000 sudah digunakan

**Solusi**: Edit `app.py` baris terakhir, ganti `port=5000` ke port lain

### Peta tidak muncul / blank

**Solusi**:

- Check folder `static/` ada file HTML map
- Cek browser console untuk error
- Pastikan Plotly terinstall dengan benar
- Clear browser cache dan refresh

### Web loading lambat

**Solusi**:

- Pastikan ARIMA cache sudah di-generate
- Tutup aplikasi lain yang berat
- Restart Flask app

## 🎓 Background Teori

### ARIMA (AutoRegressive Integrated Moving Average)

Model statistik untuk prediksi time series dengan tiga komponen:

- **AR (p)**: Autoregressive - prediksi berdasarkan nilai sebelumnya
- **I (d)**: Integrated - differencing untuk stationarity
- **MA (q)**: Moving Average - prediksi berdasarkan error sebelumnya

### K-Means Clustering

Algoritma unsupervised learning untuk mengelompokkan data ke dalam K clusters berdasarkan kesamaan fitur, dengan meminimalkan within-cluster sum of squares.

### ARIMA + K-Means Hybrid

Kombinasi inovatif:

1. ARIMA memprediksi nilai masa depan
2. K-Means melakukan re-clustering berdasarkan prediksi tersebut
3. Menghasilkan insight tentang perubahan pola di masa depan

### LSTM Autoencoder

Deep learning architecture untuk:

- Ekstraksi fitur non-linear dari time series
- Dimensionality reduction ke latent space
- Pattern recognition temporal

### N-BEATS (Neural Basis Expansion Analysis)

State-of-the-art neural network untuk time series forecasting dengan:

- Interpretable architecture
- No need for feature engineering
- Competitive dengan statistical models

## 📧 Pengembangan Lebih Lanjut

Aplikasi ini dapat dikembangkan dengan:

### Machine Learning

- Upload file CSV secara dinamis
- Parameter clustering yang dapat disesuaikan (jumlah cluster, algorithm)
- Multiple clustering algorithms comparison (K-Means, DBSCAN, Hierarchical)
- ARIMA parameter tuning interface
- Export hasil visualisasi ke PDF/Excel

### Deep Learning

- Transfer learning untuk negara dengan data limited
- Ensemble models (combine ARIMA + N-BEATS)
- Multi-variate forecasting (incorporate GDP, population, etc)

### Web Features

- User authentication & personalization
- Dashboard customization
- API endpoint RESTful untuk integrasi
- Mobile app integration

---

## 🌟 Fitur Highlight: ARIMA + K-Means Hybrid

### Mengapa Fitur Ini Penting?

1. **Predictive Analytics**: Tidak hanya melihat pola masa lalu, tapi juga masa depan
2. **Dynamic Clustering**: Cluster berubah sesuai prediksi, bukan data historis statis
3. **Policy Making**: Membantu stakeholder mengidentifikasi negara high-risk di masa depan
4. **Scalability**: Cache system membuat aplikasi scalable untuk banyak user
5. **User Control**: Flexibility untuk eksplorasi berbagai time horizon

---

**Dibuat untuk**: Analisis Big Data - UAS Semester 7
**Framework**: Flask 3.0
**Python**: 3.8+
**Last Updated**: 2024

## 📚 Referensi

- [Flask Documentation](https://flask.palletsprojects.com/)
- [pmdarima Documentation](https://alkaline-ml.com/pmdarima/)
- [Plotly Python](https://plotly.com/python/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

## 📖 Panduan Lengkap

Untuk panduan lengkap fitur ARIMA + K-Means Hybrid, baca file:
**`ARIMA_KMEANS_GUIDE.md`**

---

💡 **Tips**: Mulai dengan dashboard (`/`), toggle antara ML dan DL untuk explore semua fitur!
