# Create a Streamlit demo app for fingerprint analysis
import json, os, textwrap, zipfile
from pathlib import Path
import io
import base64
import json
import numpy as np
import cv2
import streamlit as st
from skimage import filters, morphology, measure, exposure
from skimage.util import img_as_ubyte
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import gaussian_filter

st.set_page_config(page_title="Demo Analisis Sidik Jari (Streamlit)", layout="wide")

st.title("🧪 Demo Analisis Sidik Jari")
st.caption("Upload gambar sidik jari → sistem akan menampilkan praproses, skeleton, dan estimasi fitur (minutiae sederhana, kualitas, koherensi).")

# --- Sidebar controls
with st.sidebar:
    st.header("Pengaturan")
    dpi = st.number_input("DPI (opsional)", min_value=100, max_value=2000, value=500, step=50, help="Jika diketahui DPI sensor, masukkan di sini untuk estimasi kepadatan ridge per mm.")
    block = st.slider("Ukuran blok orientasi (px)", 8, 64, 16, step=4)
    thr = st.slider("Ambang segmentasi (0-1)", 0.1, 0.9, 0.35, step=0.01)
    min_obj = st.slider("Minimum area objek (px)", 32, 2000, 400, step=16)
    show_vectors = st.checkbox("Tampilkan vektor orientasi (lebih lambat)", value=False)
    st.markdown("---")
    st.info("Tips: gunakan citra grayscale 500dpi untuk hasil terbaik.")

uploaded = st.file_uploader("Unggah gambar sidik jari (JPG/PNG/BMP)", type=["jpg", "jpeg", "png", "bmp"])

def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def normalize(img):
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (3,3), 0)
    return img

def focus_measure(gray):
    # Variance of Laplacian: semakin tinggi, semakin tajam
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(fm)

def estimate_orientation(gray, blk):
    # Sobel gradients
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    Gxx = gaussian_filter(gx*gx, sigma=blk/6)
    Gyy = gaussian_filter(gy*gy, sigma=blk/6)
    Gxy = gaussian_filter(gx*gy, sigma=blk/6)
    # Orientation (half-angle)
    theta = 0.5 * np.arctan2(2*Gxy, (Gxx - Gyy + 1e-8))
    # Coherence (0..1)
    lam1 = (Gxx + Gyy)/2 + np.sqrt(((Gxx - Gyy)/2)**2 + Gxy**2)
    lam2 = (Gxx + Gyy)/2 - np.sqrt(((Gxx - Gyy)/2)**2 + Gxy**2)
    coherence = (lam1 - lam2) / (lam1 + lam2 + 1e-8)
    return theta, np.clip(coherence, 0, 1)

def segment_fingerprint(gray, thr=0.35, min_area=400):
    # CLAHE untuk kontras ridge
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    # normalisasi 0..1
    g01 = (g - g.min()) / (g.max() - g.min() + 1e-8)
    # threshold adaptif sederhana
    mask = (g01 > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=2)
    # remove small objects
    labeled = measure.label(mask, connectivity=2)
    props = measure.regionprops(labeled)
    keep = np.zeros_like(mask)
    for p in props:
        if p.area >= min_area:
            keep[labeled==p.label] = 1
    return (g, keep.astype(bool))

def enhance_ridges(gray):
    # Gabor filter bank sederhana pada beberapa orientasi
    thetas = np.linspace(0, np.pi, 8, endpoint=False)
    accum = np.zeros_like(gray, dtype=np.float32)
    for th in thetas:
        ksize = 15
        sigma = 4.0
        lambd = 8.0
        gamma = 0.5
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, th, lambd, gamma, 0, ktype=cv2.CV_32F)
        f = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
        accum = np.maximum(accum, f)
    accum = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return accum

def binarize_and_skeletonize(img, mask):
    # Otsu pada area bertopeng
    masked = img.copy()
    masked[~mask] = 0
    th_val, bw = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bw = (bw > 0) & mask
    # thinning/skeleton
    skel = skeletonize(bw)
    skel = remove_small_objects(skel, min_size=30)
    return bw.astype(np.uint8)*255, skel

def crossing_number(window):
    # window 3x3 boolean, pusat [1,1]
    p = window.astype(np.uint8)
    # urutan tetangga 8-arah searah jarum jam
    n = [p[0,1], p[0,2], p[1,2], p[2,2], p[2,1], p[2,0], p[1,0], p[0,0], p[0,1]]
    cn = 0
    for i in range(len(n)-1):
        cn += (n[i] == 0 and n[i+1] == 1)
    return cn

def detect_minutiae(skel):
    pts_end = []
    pts_bif = []
    s = skel.astype(np.uint8)
    h, w = s.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if s[y,x] == 1:
                win = s[y-1:y+2, x-1:x+2]
                cn = crossing_number(win)
                if cn == 1:
                    pts_end.append((x,y))
                elif cn == 3:
                    pts_bif.append((x,y))
    return np.array(pts_end), np.array(pts_bif)

def ridges_per_mm(orientation_coherence, dpi):
    # Perkiraan kasar kepadatan ridge berdasarkan frekuensi spasial dari FFT global
    # (sederhana untuk demo; gunakan blok FFT per-area untuk produksi)
    # Normalisasi
    arr = orientation_coherence
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    # gunakan varian sebagai proxy "ketegasan ridge", bukan frekuensi sesungguhnya
    proxy = float(arr.mean())
    # heuristik: rata-rata jarak ridge ~ 0.4 mm pada 500dpi ⇒ ~ 8 px per ridge
    # konversi px ke mm: px_per_mm = dpi / 25.4
    px_per_mm = dpi / 25.4
    ridges_per_mm_est = px_per_mm / 8.0  # ~ perkiraan
    return float(max(0.1, min(10.0, ridges_per_mm_est))), proxy

def overlay_points(img_rgb, pts, color):
    out = img_rgb.copy()
    for (x,y) in pts:
        cv2.circle(out, (int(x),int(y)), 3, color, -1, lineType=cv2.LINE_AA)
    return out

def make_report(meta):
    return json.dumps(meta, indent=2)

if uploaded is None:
    st.info("Silakan unggah gambar sidik jari untuk memulai analisis.")
    st.stop()

# --- Read & preprocess
file_bytes = np.frombuffer(uploaded.read(), np.uint8)
bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if bgr is None:
    st.error("Gagal membaca gambar. Pastikan format file valid.")
    st.stop()

gray = to_gray(bgr)
gray = cv2.resize(gray, dsize=None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
norm = normalize(gray)

# --- Quality & orientation
theta, coherence = estimate_orientation(norm, block)
fm = focus_measure(norm)
coh_mean = float(np.mean(coherence))

# --- Segmentation & enhancement
enh_in, mask = segment_fingerprint(norm, thr=thr, min_area=min_obj)
enh = enhance_ridges(enh_in)
bw, skel = binarize_and_skeletonize(enh, mask)

# --- Minutiae (CN-based)
ends, bifs = detect_minutiae(skel)

# --- Ridge density (heuristic)
ridges_mm, flow_proxy = ridges_per_mm(coherence, dpi=dpi)

# --- Compose outputs
col1, col2, col3 = st.columns(3)
col1.subheader("Asli")
col1.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
col2.subheader("Praproses & Peningkatan")
col2.image(enh, clamp=True, use_column_width=True)
col3.subheader("Skeleton + Minutiae")
rgb = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)
rgb = overlay_points(rgb, ends, (0,255,0))
rgb = overlay_points(rgb, bifs, (255,0,0))
col3.image(rgb, channels="RGB", use_column_width=True, caption="Hijau: Ridge endings, Merah: Bifurcations (estimasi)")

# --- Orientation vectors (optional)
if show_vectors:
    import matplotlib.pyplot as plt
    st.subheader("Bidang Orientasi (preview)")
    step = block
    Y, X = np.mgrid[step//2:gray.shape[0]:step, step//2:gray.shape[1]:step]
    Q = plt.figure()
    plt.imshow(gray, cmap='gray')
    U = np.cos(theta[step//2::step, step//2::step])
    V = np.sin(theta[step//2::step, step//2::step])
    plt.quiver(X, Y, U, -V, color='r', pivot='mid', scale=30)
    st.pyplot(Q)

# --- Metrics / data
st.markdown("### 📊 Ringkasan Data Sidik Jari")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Focus Measure (Laplacian)", f"{fm:.1f}")
c2.metric("Koherensi rata-rata", f"{coh_mean:.3f}")
c3.metric("Ridge endings (estimasi)", f"{len(ends)}")
c4.metric("Bifurcations (estimasi)", f"{len(bifs)}")

st.markdown("---")
st.markdown("#### Estimasi Kepadatan Ridge")
st.write(f"Perkiraan ridges per mm (heuristik): **{ridges_mm:.2f}**  \nProxy kelancaran aliran ridge (0..1): **{flow_proxy:.3f}**")

# --- JSON report
report = {
    "image_shape": bgr.shape[:2],
    "dpi": dpi,
    "focus_measure": fm,
    "coherence_mean": coh_mean,
    "ridge_endings_est": int(len(ends)),
    "bifurcations_est": int(len(bifs)),
    "ridges_per_mm_est": ridges_mm,
    "flow_proxy": flow_proxy,
    "notes": "Minutiae dan kepadatan ridge bersifat estimasi untuk keperluan demo; gunakan algoritma khusus untuk produksi."
}
st.markdown("#### Unduh Laporan (JSON)")
st.download_button("Download JSON", data=make_report(report), file_name="fingerprint_report.json", mime="application/json")

st.markdown("---")
with st.expander("Catatan Teknis"):
    st.write("""
- **Segmentasi** menggunakan CLAHE + threshold + morfologi, lalu skeletonize.
- **Minutiae** dihitung dengan **Crossing Number** pada skeleton (hijau: endings, merah: bifurcation). Ini **pendekatan demo**; hasil produksi perlu penapisan/matching lokal untuk menurunkan false positives.
- **Orientasi & Koherensi** memakai turunan Sobel + struktur tensor yang dihaluskan.
- **Ridges per mm** adalah **heuristik**; untuk akurasi gunakan estimasi frekuensi ridge per-blok via FFT dan kalibrasi DPI.
""")