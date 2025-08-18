# ​ Streamlit Fingerprint Analysis Demo

Aplikasi demo berbasis **Streamlit** untuk analisis sidik jari.  
Pengguna dapat mengunggah citra sidik jari (JPG/PNG/BMP), kemudian sistem akan melakukan preprocessing, peningkatan kontras, skeletonisasi, dan menampilkan estimasi fitur dasar seperti **minutiae (ridge endings & bifurcations)** serta beberapa metrik kualitas.

---

##  Repositori
Temukan kode sumber dan dokumentasi di GitHub:  
🔗 [https://github.com/ardasurya/fingerprintanalysisai.git](https://github.com/ardasurya/fingerprintanalysisai.git)

---

##  Fitur
- Upload gambar sidik jari.
- Preprocessing otomatis (CLAHE + Gabor filter bank).
- Segmentasi area sidik jari dan skeletonisasi.
- Deteksi minutiae sederhana dengan **Crossing Number**:
  - Hijau = Ridge endings
  - Merah = Bifurcations
- Estimasi kualitas:
  - Focus Measure (Variance of Laplacian)
  - Koherensi rata-rata orientasi ridge
  - Estimasi ridge per mm (berdasarkan DPI)
- Ringkasan data dapat diunduh dalam format **JSON**.

---

##  Instalasi
1. Clone repositori:
   ```bash
   git clone https://github.com/ardasurya/fingerprintanalysisai.git
   cd fingerprintanalysisai
