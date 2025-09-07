import numpy as np
import cv2
import streamlit as st
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize, remove_small_objects
from skimage import measure
from skimage.draw import line as draw_line
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Fingerprint AI Demo – Core/Delta, Pattern & Ridge Count", layout="wide")
st.title("🌀 Fingerprint AI Demo – Core/Delta, Pattern & Ridge Count (dengan Core Draggable)")
st.caption("Upload sidik jari → deteksi core/delta, klasifikasi pola, ridge count core→delta & radial, minutiae, metrik kualitas. Bisa geser titik core secara manual.")

# ================= Sidebar =================
with st.sidebar:
    st.header("Pengaturan")
    dpi = st.number_input("DPI (opsional)", 100, 2000, 500, step=50,
                          help="Jika tidak tahu, biarkan 500 (umum untuk sensor).")
    block = st.slider("Ukuran blok orientasi (px)", 8, 64, 24, step=4)
    thr = st.slider("Ambang segmentasi (0–1)", 0.1, 0.9, 0.35, step=0.01)
    min_obj = st.slider("Minimum area objek (px)", 32, 4000, 600, step=20)
    n_rays = st.slider("Jumlah sinar radial", 8, 120, 36, step=2)
    max_radius = st.slider("Radius maksimum (px)", 50, 1000, 300, step=10)
    show_vectors = st.checkbox("Tampilkan bidang orientasi (preview)", value=False)
    show_minutiae = st.checkbox("Tampilkan titik minutiae (estimasi)", value=True)
    st.markdown("---")
    use_manual_core = st.checkbox(
        "Gunakan core manual (draggable)",
        value=False,
        help="Aktifkan untuk memilih/menggeser core secara manual di kanvas."
    )
    st.info("Tips: gunakan citra whorl/loop grayscale 500 dpi. Coba block 24–32 agar core/delta stabil.")

uploaded = st.file_uploader("Unggah gambar sidik jari (JPG/PNG/BMP)", type=["jpg","jpeg","png","bmp"])

# ================= Util =================
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def normalize(img):
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (3,3), 0)
    return img

def focus_measure(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def estimate_orientation(gray, blk):
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    Gxx = gaussian_filter(gx*gx, sigma=blk/6)
    Gyy = gaussian_filter(gy*gy, sigma=blk/6)
    Gxy = gaussian_filter(gx*gy, sigma=blk/6)
    theta = 0.5 * np.arctan2(2*Gxy, (Gxx - Gyy + 1e-8))
    lam1 = (Gxx + Gyy)/2 + np.sqrt(((Gxx - Gyy)/2)**2 + Gxy**2)
    lam2 = (Gxx + Gyy)/2 - np.sqrt(((Gxx - Gyy)/2)**2 + Gxy**2)
    coherence = (lam1 - lam2) / (lam1 + lam2 + 1e-8)
    # smoothing orientasi via cos(2θ)/sin(2θ)
    c2 = np.cos(2*theta); s2 = np.sin(2*theta)
    c2 = gaussian_filter(c2, sigma=blk/3)
    s2 = gaussian_filter(s2, sigma=blk/3)
    theta_smooth = 0.5 * np.arctan2(s2, c2)
    coherence_smooth = gaussian_filter(coherence, sigma=blk/3)
    return theta_smooth, np.clip(coherence_smooth, 0, 1)

def segment_fingerprint(gray, thr=0.35, min_area=600):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    g01 = (g - g.min()) / (g.max() - g.min() + 1e-8)
    mask = (g01 > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 2)
    labeled = measure.label(mask, connectivity=2)
    keep = np.zeros_like(mask)
    for p in measure.regionprops(labeled):
        if p.area >= min_area:
            keep[labeled == p.label] = 1
    return g, keep.astype(bool)

def enhance_ridges(gray):
    thetas = np.linspace(0, np.pi, 8, endpoint=False)
    accum = np.zeros_like(gray, dtype=np.float32)
    for th in thetas:
        kernel = cv2.getGaborKernel((15,15), 4.0, th, 8.0, 0.5, 0, ktype=cv2.CV_32F)
        f = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
        accum = np.maximum(accum, f)
    accum = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return accum

def binarize_and_skeletonize(img, mask):
    masked = img.copy(); masked[~mask] = 0
    _, bw = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = (bw > 0) & mask
    skel = skeletonize(bw)
    skel = remove_small_objects(skel, min_size=40)
    return (bw.astype(np.uint8) * 255), skel

# ---------- Poincaré index di grid ----------
def poincare_index_map(theta, blk):
    step = blk
    grid = theta[step//2::step, step//2::step]
    gh, gw = grid.shape
    idx_map = np.zeros_like(grid)
    for y in range(1, gh-1):
        for x in range(1, gw-1):
            w = grid[y-1:y+2, x-1:x+2]
            seq = [w[0,1], w[0,2], w[1,2], w[2,2], w[2,1], w[2,0], w[1,0], w[0,0], w[0,1]]
            diffs = []
            for i in range(len(seq)-1):
                d = seq[i+1] - seq[i]
                while d >  np.pi/2: d -= np.pi
                while d < -np.pi/2: d += np.pi
                diffs.append(d)
            idx_map[y, x] = np.sum(diffs)
    return idx_map

def core_delta_candidates(theta, coherence, blk, mask, min_abs=1.9, want_positive=True, topk=3):
    idx_map = poincare_index_map(theta, blk)
    step = blk
    cand = []
    for (y,x), val in np.ndenumerate(idx_map):
        if (abs(val) >= min_abs) and ((val > 0) == want_positive):
            py = int(y*step + step//2)
            px = int(x*step + step//2)
            if 0 <= py < coherence.shape[0] and 0 <= px < coherence.shape[1] and mask[py, px]:
                cand.append((float(coherence[py, px]), (px, py), float(val)))
    cand.sort(reverse=True)
    return cand[:topk]

# ---------- Minutiae (estimasi) ----------
def crossing_number(window3):
    p = window3.astype(np.uint8)
    n = [p[0,1], p[0,2], p[1,2], p[2,2], p[2,1], p[2,0], p[1,0], p[0,0], p[0,1]]
    cn = 0
    for i in range(len(n)-1):
        cn += (n[i] == 0 and n[i+1] == 1)
    return cn

def detect_minutiae(skel):
    s = skel.astype(np.uint8)
    h, w = s.shape
    ends, bifs = [], []
    for y in range(1, h-1):
        for x in range(1, w-1):
            if s[y, x] == 1:
                cn = crossing_number(s[y-1:y+2, x-1:x+2])
                if cn == 1: ends.append((x, y))
                elif cn == 3: bifs.append((x, y))
    return np.array(ends), np.array(bifs)

# ---------- Ridge count garis & radial ----------
def count_crossings_on_line(skel, p0, p1):
    (x0, y0), (x1, y1) = (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1]))
    rr, cc = draw_line(y0, x0, y1, x1)
    h, w = skel.shape
    prev = 0; cnt = 0
    for y, x in zip(rr, cc):
        if 0 <= x < w and 0 <= y < h:
            val = 1 if skel[y, x] else 0
            if prev == 0 and val == 1:
                cnt += 1
            prev = val
    return cnt, (rr, cc)

def count_crossings_from_center(skel, center, n_rays=36, max_radius=300):
    h, w = skel.shape
    cx, cy = int(center[0]), int(center[1])
    angs = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
    counts, rays_pts = [], []
    for a in angs:
        x2 = int(cx + max_radius*np.cos(a))
        y2 = int(cy + max_radius*np.sin(a))
        rr, cc = draw_line(cy, cx, y2, x2)
        prev = 0; c = 0; ray_pts = []
        for y, x in zip(rr, cc):
            if 0 <= x < w and 0 <= y < h:
                val = 1 if skel[y, x] else 0
                ray_pts.append((x, y, val))
                if prev == 0 and val == 1:
                    c += 1
                prev = val
        counts.append(c)
        rays_pts.append(ray_pts)
    return np.array(counts, dtype=int), rays_pts

# ---------- Klasifikasi pola (heuristik) ----------
def classify_pattern(core_pts, delta_pts, img_shape, hand=None):
    n_core = len(core_pts)
    n_delta = len(delta_pts)
    H, W = img_shape

    if n_delta >= 2:
        if n_core >= 2:
            return "Whorl – kemungkinan Double Loop (heuristik)"
        else:
            return "Whorl – Plain/Central Pocket (heuristik)"
    elif n_delta == 1:
        core = core_pts[0] if n_core else (W//2, H//2)
        d = delta_pts[0]
        dir_rel = "kanan" if d[0] > core[0] else "kiri"
        return f"Loop (delta relatif di sisi {dir_rel} core)"
    else:
        return "Arch (kemungkinan tidak ada core/delta jelas)"

# ---------- Visual overlay ----------
def overlay_all(base_gray, core, deltas, cd_lines, rays_pts, ends, bifs, draw_minutiae=True):
    rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2RGB)
    if core is not None:
        cv2.circle(rgb, (int(core[0]), int(core[1])), 6, (0,255,255), -1, lineType=cv2.LINE_AA)
    for (pt, val) in deltas:
        cv2.circle(rgb, (int(pt[0]), int(pt[1])), 6, (255,0,0), -1, lineType=cv2.LINE_AA)
    for rr, cc in cd_lines:
        rgb[rr, cc] = (255, 200, 0)  # core→delta
    for ray in rays_pts:
        for (x,y,val) in ray[::8]:
            rgb[y, x] = (0,255,0) if val==1 else (255,255,255)  # sinar radial
    if draw_minutiae and len(ends):
        for (x,y) in ends:
            cv2.circle(rgb, (int(x), int(y)), 2, (0,200,0), -1)
    if draw_minutiae and len(bifs):
        for (x,y) in bifs:
            cv2.circle(rgb, (int(x), int(y)), 2, (255,0,0), -1)
    return rgb

# ================= Main =================
if uploaded is None:
    st.info("Silakan unggah gambar sidik jari untuk memulai analisis.")
    st.stop()

file_bytes = np.frombuffer(uploaded.read(), np.uint8)
bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if bgr is None:
    st.error("Gagal membaca gambar.")
    st.stop()

gray = to_gray(bgr)
norm = normalize(gray)

# Orientasi, kualitas, segmentasi
theta, coherence = estimate_orientation(norm, block)
fm = focus_measure(norm)
enh_in, mask = segment_fingerprint(norm, thr=thr, min_area=min_obj)

# Enhancement, skeleton, minutiae
enh = enhance_ridges(enh_in)
masked = enh.copy(); masked[~mask] = 0
_, bw = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
skel = skeletonize((bw > 0) & mask)
skel = remove_small_objects(skel, min_size=40)
ends, bifs = detect_minutiae(skel)

# ---- Kanvas untuk core manual (opsional, dengan lingkaran default terlihat) ----
manual_core_xy = None
if use_manual_core:
    st.subheader("🎯 Pilih / Geser Titik Core (Manual)")

    # Siapkan background
    enh_rgb = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)
    bg_pil = Image.fromarray(enh_rgb)

    # Siapkan lingkaran awal di posisi core otomatis (kalau ada)
    default_radius = 28
    initial = None
    if 'auto_core' in locals() and auto_core is not None:
        initial = {
            "version": "5.2.4",
            "objects": [{
                "type": "circle",
                "left": float(auto_core[0] - default_radius),
                "top":  float(auto_core[1] - default_radius),
                "radius": float(default_radius),
                "fill": "rgba(255,255,0,0.25)",
                "stroke": "#ffff00",
                "strokeWidth": 3
            }]
        }

    # Bila sudah ada lingkaran awal -> mode transform; kalau belum -> mode circle
    drawing_mode = "transform" if initial is not None else "circle"

    canvas = st_canvas(
        fill_color="rgba(255,255,0,0.25)",  # isi lingkaran
        stroke_width=3,
        stroke_color="#ffff00",             # outline kuning tebal
        background_color="#00000000",       # transparan (pakai image sebagai background)
        background_image=bg_pil,            # tampilkan citra sebagai latar
        update_streamlit=True,
        height=enh_rgb.shape[0],
        width=enh_rgb.shape[1],
        drawing_mode=drawing_mode,          # transform jika sudah ada circle awal
        display_toolbar=True,
        key="core_canvas",
        initial_drawing=initial             # <<-- inilah yang memunculkan lingkaran default
    )

    if drawing_mode == "circle":
        st.info("Klik ikon **Circle** di toolbar, buat satu lingkaran di pusat pola, lalu gunakan tool panah untuk **geser**.")

    # Ambil koordinat pusat dari objek lingkaran di kanvas
    if canvas and canvas.json_data is not None:
        objs = canvas.json_data.get("objects", [])
        circle_obj = None
        for obj in objs[::-1]:
            if obj.get("type") == "circle":
                circle_obj = obj
                break
        if circle_obj is not None:
            left = float(circle_obj.get("left", 0.0))
            top = float(circle_obj.get("top", 0.0))
            radius = float(circle_obj.get("radius", 0.0))
            cx = left + radius
            cy = top + radius
            manual_core_xy = (int(round(cx)), int(round(cy)))

    if manual_core_xy is not None:
        st.success(f"Core manual: (x, y) = {manual_core_xy[0]}, {manual_core_xy[1]}")
    else:
        st.warning("Belum ada lingkaran. Buat lingkaran (mode Circle), lalu geser dengan tool panah (Transform).")

# --- Deteksi core & delta otomatis (tetap) ---
core_cands = core_delta_candidates(theta, coherence, block, mask, want_positive=True, topk=2)
delta_cands = core_delta_candidates(theta, coherence, block, mask, want_positive=False, topk=3)

auto_core = core_cands[0][1] if core_cands else None
auto_core_pi = core_cands[0][2] if core_cands else 0.0

# Pakai core manual jika ada; kalau tidak, pakai otomatis
if use_manual_core and (manual_core_xy is not None):
    core = manual_core_xy
    core_pi = 0.0
else:
    core = auto_core
    core_pi = auto_core_pi


# --- Deteksi core & delta otomatis ---
core_cands = core_delta_candidates(theta, coherence, block, mask, want_positive=True, topk=2)
delta_cands = core_delta_candidates(theta, coherence, block, mask, want_positive=False, topk=3)

auto_core = core_cands[0][1] if core_cands else None
auto_core_pi = core_cands[0][2] if core_cands else 0.0

# Gunakan core manual jika tersedia
if use_manual_core and (manual_core_xy is not None):
    core = manual_core_xy
    core_pi = 0.0
else:
    core = auto_core
    core_pi = auto_core_pi

deltas = [(c[1], c[2]) for c in delta_cands]  # list of (pt, poincare_neg)

# Ridge count core→delta
cd_counts = []
cd_lines = []
if core is not None and len(deltas) > 0:
    for (dpt, dval) in deltas:
        cnt, (rr, cc) = count_crossings_on_line(skel, core, dpt)
        cd_counts.append({"delta_xy": (int(dpt[0]), int(dpt[1])), "poincare": float(dval), "ridge_count": int(cnt)})
        cd_lines.append((rr, cc))

# Ridge count radial
rad_counts, rays_pts = None, []
if core is not None:
    rad_counts, rays_pts = count_crossings_from_center(skel.astype(np.uint8), core, n_rays=n_rays, max_radius=max_radius)

# Klasifikasi pola (heuristik)
core_pts = [core] if core is not None else []
delta_pts = [d[0] for d in deltas]
pattern_label = classify_pattern(core_pts, delta_pts, skel.shape)

# Luas ROI + estimasi ridges/mm
roi_area_px = int(mask.sum())
px_per_mm = dpi / 25.4
area_mm2 = roi_area_px / (px_per_mm**2 + 1e-8)
ridges_per_mm_est = float(np.clip(px_per_mm / 8.0, 0.1, 12.0))
coh_mean = float(np.mean(coherence[mask])) if mask.any() else 0.0

# ================= Layout =================
c1, c2, c3 = st.columns(3)
c1.subheader("Asli")
c1.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

c2.subheader("Enhancement")
c2.image(enh, clamp=True, use_column_width=True)

# Overlay
cd_lines_viz = [ln for ln in cd_lines]
viz = overlay_all(enh, core, deltas, cd_lines_viz, rays_pts, ends, bifs, draw_minutiae=show_minutiae)
c3.subheader("Skeleton + Overlay (Core/Delta/Rays/Minutiae)")
cap = "Kuning: Core • Merah: Delta • Oranye: Core→Delta • Hijau: sinar radial • Hijau tua: endings • Merah tua: bifurcations"
c3.image(viz, channels="RGB", use_column_width=True, caption=cap)

st.markdown("### 📊 Ringkasan & Informasi Sidik Jari")
info_cols = st.columns(4)
info_cols[0].metric("Focus (Laplacian)", f"{fm:.1f}")
info_cols[1].metric("Koherensi rata-rata", f"{coh_mean:.3f}")
info_cols[2].metric("ROI area (px²)", f"{roi_area_px}")
info_cols[3].metric("ROI area (mm², est.)", f"{area_mm2:.1f}")

st.write(f"**Jenis Pola (heuristik):** {pattern_label}")
if core is not None:
    st.write(f"**Core (x,y):** {int(core[0])}, {int(core[1])} | **Poincaré(core):** {core_pi:.2f} {'(manual)' if use_manual_core and manual_core_xy else ''}")
else:
    st.warning("Core tidak terdeteksi meyakinkan. Coba perbesar `block` (24–32) dan pastikan pola whorl/loop jelas.")

if deltas:
    st.write(f"**Jumlah Delta terdeteksi:** {len(deltas)} (Poincaré ≈ −π).")
    for i, (pt, val) in enumerate(deltas, 1):
        dist = float(np.hypot(pt[0]-core[0], pt[1]-core[1])) if core is not None else 0.0
        st.write(f"- Delta #{i} @ ({int(pt[0])},{int(pt[1])}) | Poincaré: {val:.2f} | Jarak ke core: {dist:.1f}px")
else:
    st.info("Delta tidak terdeteksi. Pada pola arch biasanya memang tidak ada; pada loop/whorl, coba atur parameter.")

# Ridge count core→delta
if cd_counts:
    st.markdown("#### Ridge Count – Core → Delta")
    for i, d in enumerate(cd_counts, 1):
        st.write(f"{i}. Delta @ {d['delta_xy']} | Poincaré: {d['poincare']:.2f} | **Ridge Count: {d['ridge_count']}**")

# Ridge radial
if rad_counts is not None and len(rad_counts):
    st.markdown("#### Ridge Count – Radial dari Core")
    rA, rB, rC = st.columns(3)
    rA.metric("Rata-rata", f"{float(np.mean(rad_counts)):.2f}")
    rB.metric("Median", f"{float(np.median(rad_counts)):.2f}")
    rC.metric("Maksimum", f"{int(np.max(rad_counts))}")
    with st.expander("Detail per-arah (array)"):
        st.code(str(rad_counts.tolist()))

# Bidang orientasi (opsional)
if show_vectors:
    import matplotlib.pyplot as plt
    st.subheader("Bidang Orientasi (preview)")
    step = block
    Y, X = np.mgrid[step//2:gray.shape[0]:step, step//2:gray.shape[1]:step]
    U = np.cos(theta[step//2::step, step//2::step])
    V = np.sin(theta[step//2::step, step//2::step])
    fig = plt.figure()
    plt.imshow(gray, cmap='gray')
    plt.quiver(X, Y, U, -V, color='r', pivot='mid', scale=30)
    if core is not None: plt.scatter([core[0]], [core[1]], c='y', s=40, label='Core')
    if deltas:
        for (pt, _v) in deltas:
            plt.scatter([pt[0]], [pt[1]], c='r', s=30)
    st.pyplot(fig)

st.markdown("---")
with st.expander("Catatan Teknis"):
    st.write("""
- **Core/Delta** dideteksi via **Poincaré index** pada peta orientasi yang sudah dihaluskan (`cos(2θ)`, `sin(2θ)` + Gaussian).
- **Core manual** dapat dipilih/geser menggunakan `streamlit-drawable-canvas` (lingkaran kuning). Jika aktif, perhitungan memakai core manual.
- **Klasifikasi pola (heuristik)**:
  - `delta ≥ 2` → Whorl (jika `core ≥ 2` → indikasi **Double Loop**; selain itu **Plain/Central-Pocket**).
  - `delta = 1` → Loop (arah relatif kiri/kanan terhadap core).
  - `delta = 0` → Arch.
- **Ridge Count**:
  - **Core→Delta**: jumlah transisi 0→1 pada skeleton di garis lurus penghubung.
  - **Radial dari Core**: jumlah persilangan pada N sinar (diagnostik tambahan).
- **Minutiae (estimasi)** dihitung dari **Crossing Number** pada skeleton; masih bersifat demo (perlu penapisan untuk menekan false positives).
- **Metrik kualitas**: Focus (Laplacian), koherensi rata-rata, luas ROI (px² & mm²), estimasi ridge density (ridges/mm).
- Produksi: sertakan **quality gating (NFIQ2)**, estimasi frekuensi ridge per-blok (FFT), dan validasi FVC/LivDet.
""")
