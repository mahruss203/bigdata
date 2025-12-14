# ARIMA + K-Means Hybrid Clustering - Panduan Penggunaan

## 📋 Deskripsi Fitur

Fitur **ARIMA + K-Means Hybrid Clustering** adalah inovasi yang menggabungkan dua metode machine learning:

- **ARIMA (AutoRegressive Integrated Moving Average)**: Untuk prediksi time series
- **K-Means Clustering**: Untuk pengelompokan negara berdasarkan nilai prediksi

## 🎯 Cara Kerja

### 1. Pre-compute ARIMA Models (Offline Process)

```bash
python generate_arima_cache.py
```

Script ini akan:

- Membaca data historis kematian stroke untuk semua negara (2000-2021)
- Melakukan training ARIMA model untuk setiap negara
- Menyimpan model ke dalam cache (arima_models_cache.pkl)
- Menghasilkan metadata (arima_cache_metadata.pkl)

**Output:**

- `arima_models_cache.pkl`: Berisi semua model ARIMA yang sudah di-train
- `arima_cache_metadata.pkl`: Metadata seperti jumlah negara, waktu update, dll

### 2. Web Application (Online Process)

Akses fitur melalui browser:

```
http://127.0.0.1:5000/arima-kmeans
```

#### Langkah-langkah penggunaan:

1. **Pilih Jumlah Tahun Prediksi**

   - Gunakan slider untuk memilih antara 1-5 tahun
   - Slider interaktif menampilkan nilai real-time

2. **Klik "Generate Prediksi & Clustering"**

   - Sistem akan memuat model ARIMA dari cache (cepat, tanpa training ulang)
   - Melakukan prediksi untuk semua negara sesuai tahun yang dipilih
   - Menghitung rata-rata prediksi per negara
   - Melakukan K-Means clustering berdasarkan nilai prediksi
   - Menampilkan peta dunia interaktif

3. **Interpretasi Hasil**
   - **Peta Dunia**: Setiap negara diwarnai berdasarkan cluster (0, 1, 2)
   - **Cluster 0 (Hijau)**: Risiko rendah - negara dengan prediksi kematian stroke rendah
   - **Cluster 1 (Kuning)**: Risiko sedang - negara dengan prediksi menengah
   - **Cluster 2 (Merah)**: Risiko tinggi - negara dengan prediksi tinggi

## 🚀 Keunggulan Sistem

### Performance Optimization

- **Caching System**: Model ARIMA di-cache sehingga web loading sangat cepat
- **No Re-training**: Tidak perlu training ulang setiap kali user akses
- **Background Processing**: Semua komputasi berat dilakukan offline

### User Experience

- **Interactive Slider**: Pilih tahun prediksi dengan mudah (1-5 tahun)
- **Real-time Visualization**: Peta dunia langsung ter-update setelah klik button
- **Responsive Design**: Tampilan optimal di desktop dan mobile
- **Informative Statistics**: Statistik lengkap per cluster

### Flexibility

- **Dynamic Re-clustering**: Clustering dilakukan real-time berdasarkan tahun yang dipilih
- **Multiple Time Horizons**: User dapat membandingkan hasil untuk 1, 2, 3, 4, atau 5 tahun
- **Scalable**: Mudah di-extend untuk menambah negara baru

## 📊 Struktur File

```
UAS/
├── generate_arima_cache.py          # Script untuk pre-compute ARIMA models
├── arima_models_cache.pkl           # Cache model ARIMA (generated)
├── arima_cache_metadata.pkl         # Metadata cache (generated)
├── app.py                           # Flask application
├── templates/
│   ├── index.html                   # Dashboard dengan link ke fitur baru
│   └── arima_kmeans.html            # Template untuk fitur hybrid
└── static/
    └── arima_kmeans_map.html        # Peta interaktif (generated)
```

## 🔧 Teknologi yang Digunakan

### Backend

- **Flask**: Web framework
- **pmdarima**: Auto ARIMA implementation
- **scikit-learn**: K-Means clustering
- **pandas**: Data manipulation
- **pickle**: Model serialization

### Frontend

- **Plotly**: Interactive world map
- **HTML5/CSS3**: Modern responsive design
- **JavaScript**: Interactive slider

## 📈 Workflow Teknis

```
[User Input: Pilih tahun 1-5]
         ↓
[Load ARIMA cache (pkl)]
         ↓
[Generate predictions untuk semua negara]
         ↓
[Calculate average prediction per country]
         ↓
[Standardize data dengan StandardScaler]
         ↓
[K-Means clustering (n_clusters=3)]
         ↓
[Generate Plotly choropleth map]
         ↓
[Display results: map + statistics]
```

## 🎨 Interpretasi Warna Cluster

| Cluster | Warna     | Kategori Risiko | Deskripsi                                      |
| ------- | --------- | --------------- | ---------------------------------------------- |
| 0       | 🟢 Hijau  | Rendah          | Negara dengan prediksi kematian stroke rendah  |
| 1       | 🟡 Kuning | Sedang          | Negara dengan prediksi menengah                |
| 2       | 🔴 Merah  | Tinggi          | Negara dengan prediksi tinggi, butuh perhatian |

## 💡 Tips Penggunaan

1. **Pertama kali setup**: Pastikan jalankan `generate_arima_cache.py` dulu
2. **Update data**: Jika data berubah, jalankan ulang script pre-compute
3. **Performance**: Cache sekali, pakai berkali-kali (very fast!)
4. **Comparison**: Coba berbagai tahun prediksi untuk melihat tren

## 🐛 Troubleshooting

### Error: "ARIMA cache belum dibuat"

**Solusi**: Jalankan `python generate_arima_cache.py` terlebih dahulu

### Web loading lambat

**Solusi**:

- Pastikan cache sudah di-generate
- Check ukuran file pkl (normal: 5-20 MB)
- Restart Flask app

### Peta tidak muncul

**Solusi**:

- Check folder `static/` ada file `arima_kmeans_map.html`
- Check browser console untuk error JavaScript
- Pastikan Plotly library terinstall

## 📞 Support

Jika ada pertanyaan atau issue, silakan:

1. Check error message di console
2. Verifikasi semua dependencies terinstall
3. Pastikan data CSV tersedia dan valid

---

**Created by**: Machine Learning Team
**Last Updated**: 2024
**Version**: 1.0
