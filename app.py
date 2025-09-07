# ================== IMPORTS ==================
import os, glob, time, json
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from skimage.morphology import skeletonize, remove_small_objects
from skimage import measure
from skimage.draw import line as draw_line
from streamlit_drawable_canvas import st_canvas

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Fingerprint AI Demo", layout="wide")
st.title("🌀 Fingerprint AI Demo – Core/Delta, Pattern, Ridge Count, & MobileNetV2")

# ================== SIDEBAR MENU ==================
st.sidebar.markdown("### 📚 Menu")
mode = st.sidebar.radio(
    "Pilih fitur:",
    ["🧪 Analisis"],
    index=0,
    help="Pilih mode aplikasi"
)

# ================== UTIL UMUM ==================
def safe_makedirs(p):
    os.makedirs(p, exist_ok=True)
    return p

def list_saved_models(models_dir="models"):
    safe_makedirs(models_dir)
    paths = []
    paths += sorted(glob.glob(os.path.join(models_dir, "*.h5")))
    paths += sorted(glob.glob(os.path.join(models_dir, "*.keras")))
    return paths

@st.cache_resource
def load_any_model(path):
    if os.path.isfile(path) and (path.endswith(".h5") or path.endswith(".keras")):
        return keras.models.load_model(path)
    else:
        raise ValueError(f"Path model tidak dikenal untuk load_model: {path}")


# ================== PIPELINE ANALISIS (fungsi2) ==================
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

def overlay_all(base_gray, core, deltas, cd_lines, rays_pts, ends, bifs, draw_minutiae=True):
    rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2RGB)
    if core is not None:
        cv2.circle(rgb, (int(core[0]), int(core[1])), 6, (0,255,255), -1, lineType=cv2.LINE_AA)
    for (pt, val) in deltas:
        cv2.circle(rgb, (int(pt[0]), int(pt[1])), 6, (255,0,0), -1, lineType=cv2.LINE_AA)
    for rr, cc in cd_lines:
        rgb[rr, cc] = (255, 200, 0)
    for ray in rays_pts:
        for (x,y,val) in ray[::8]:
            rgb[y, x] = (0,255,0) if val==1 else (255,255,255)
    if draw_minutiae and len(ends):
        for (x,y) in ends:
            cv2.circle(rgb, (int(x), int(y)), 2, (0,200,0), -1)
    if draw_minutiae and len(bifs):
        for (x,y) in bifs:
            cv2.circle(rgb, (int(x), int(y)), 2, (255,0,0), -1)
    return rgb

# ================== DATASET & MODEL (Training/Klasifikasi) ==================
def build_datasets(data_root="dataset", img_size=(224, 224), batch_size=32, seed=42):
    train_dir = os.path.join(data_root, "train_set")
    val_dir   = os.path.join(data_root, "val_set")
    test_dir  = os.path.join(data_root, "test_set")

    for d in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Folder tidak ditemukan: {d}")

    # 1) TRAIN raw → ambil class_names sebelum prefetch
    train_ds_raw = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=img_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
    )
    class_names = list(train_ds_raw.class_names)

    # 2) VAL/TEST: pastikan urutan label konsisten
    val_ds_raw = keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=img_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        class_names=class_names,  # <-- perbaikannya di sini
    )
    test_ds_raw = keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        class_names=class_names,  # <-- dan di sini
    )

    # 3) Prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds_raw.shuffle(1024).prefetch(AUTOTUNE)
    val_ds   = val_ds_raw.prefetch(AUTOTUNE)
    test_ds  = test_ds_raw.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names



def build_mobilenet_model(num_classes, img_size=(224,224), base_trainable=False, lr=1e-3):
    inputs = layers.Input(shape=img_size + (3,))
    x = layers.Rescaling(1./255)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    base = keras.applications.MobileNetV2(
        input_shape=img_size + (3,), include_top=False, weights="imagenet", pooling=None
    )
    base.trainable = base_trainable
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def unfreeze_and_compile(model, base_lr=1e-4):
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name.startswith("mobilenetv2"):
            base_model = layer
            break
    if base_model is None:
        return model
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(base_lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_mobilenet(data_root="dataset", img_size=(224,224), batch_size=32, epochs=10, lr=1e-3,
                    fine_tune=False, fine_tune_epochs=5, models_dir="models"):
    safe_makedirs(models_dir)
    train_ds, val_ds, test_ds, class_names = build_datasets(data_root, img_size, batch_size)
    model = build_mobilenet_model(len(class_names), img_size, base_trainable=False, lr=lr)
    ckpt_name = f"mobilenet_fp_{int(time.time())}"

    # --- Siapkan path kedua format ---
    ckpt_path_h5     = os.path.join(models_dir, ckpt_name + ".h5")
    ckpt_path_keras  = os.path.join(models_dir, ckpt_name + ".keras")
    final_path_h5    = os.path.join(models_dir, ckpt_name + "_final.h5")
    final_path_keras = os.path.join(models_dir, ckpt_name + "_final.keras")

    # --- Callback: coba pakai .h5, kalau Keras menolak -> fallback .keras ---
    try:
        ckpt_cb = keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path_h5,          # .h5 full model
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        )
        # uji early init (beberapa build keras validasi path saat konstruktor)
        _ = ckpt_cb
        best_ckpt_path = ckpt_path_h5
    except Exception:
        ckpt_cb = keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path_keras,       # fallback .keras
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        )
        best_ckpt_path = ckpt_path_keras

    cbs = [
        ckpt_cb,
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]

    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs)

    # Fine-tune (opsional)
    ft_hist = None
    if fine_tune and fine_tune_epochs > 0:
        model = unfreeze_and_compile(model, base_lr=lr/10.0)
        ft_hist = model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs, callbacks=cbs)

    # Evaluate on test
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    # --- Simpan final model: coba .h5, kalau gagal -> .keras ---
    final_saved_path = None
    try:
        model.save(final_path_h5)          # full model ke HDF5
        final_saved_path = final_path_h5
    except Exception:
        model.save(final_path_keras)       # fallback ke .keras
        final_saved_path = final_path_keras

    # (opsional) simpan label
    labels_json = os.path.join(models_dir, ckpt_name + "_labels.json")
    with open(labels_json, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    return {
        "model": model,
        "class_names": class_names,
        "hist": hist.history,
        "ft_hist": (ft_hist.history if ft_hist is not None else None),

        # format baru
        "best_model": best_ckpt_path,      # .h5 atau .keras
        "final_model": final_saved_path,   # .h5 atau .keras

        # alias untuk kompatibilitas lama (boleh dipertahankan sementara)
        "best_h5": best_ckpt_path if best_ckpt_path.endswith(".h5") else None,
        "final_h5": final_saved_path if final_saved_path and final_saved_path.endswith(".h5") else None,

        "labels_json": labels_json,
        "test_metrics": {"loss": float(test_loss), "accuracy": float(test_acc)},
    }



def plot_history(hist_dict, title="Training History"):
    fig = plt.figure()
    if "accuracy" in hist_dict: plt.plot(hist_dict["accuracy"], label="train_acc")
    if "val_accuracy" in hist_dict: plt.plot(hist_dict["val_accuracy"], label="val_acc")
    if "loss" in hist_dict: plt.plot(hist_dict["loss"], label="train_loss")
    if "val_loss" in hist_dict: plt.plot(hist_dict["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.title(title); plt.legend()
    st.pyplot(fig)
    
# ===== Helper tampilan hasil training (aman utk .keras/.h5/SavedModel) =====
def show_training_artifacts(artifacts: dict):
    best_path   = artifacts.get("best_model") or artifacts.get("best_h5")
    final_path  = artifacts.get("final_model") or artifacts.get("final_h5")
    labels_path = artifacts.get("labels_json")
    saved_dir   = artifacts.get("saved_model_dir")   # bisa None kalau tidak export
    class_names = artifacts.get("class_names", [])
    test_metrics = artifacts.get("test_metrics") or {}
    hist = artifacts.get("hist")
    ft_hist = artifacts.get("ft_hist")

    st.success("Training selesai! Hasil tersimpan.")

    cA, cB = st.columns(2)
    with cA:
        if best_path:
            st.write("**Best model:**")
            st.code(str(best_path))
        if labels_path:
            st.write("**Labels JSON:**")
            st.code(str(labels_path))
    with cB:
        if final_path:
            st.write("**Final model:**")
            st.code(str(final_path))
        if saved_dir:
            st.write("**SavedModel (export):**")
            st.code(str(saved_dir))

    if class_names:
        st.write("**Kelas (urut label):**", class_names)
    if test_metrics:
        st.write("**Test metrics:**", test_metrics)

    # Plot kurva jika tersedia
    if isinstance(hist, dict) and hist:
        plot_history(hist, title="Tahap 1 (Freeze)")
    if isinstance(ft_hist, dict) and ft_hist:
        plot_history(ft_hist, title="Tahap 2 (Fine-tune)")


def evaluate_on_test(model, test_ds, class_names):
    y_true, y_pred = [], []
    for batch_imgs, batch_labels in test_ds:
        preds = model.predict(batch_imgs, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(batch_labels.numpy(), axis=1))
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix"); plt.colorbar()
    tick = np.arange(len(class_names))
    plt.xticks(tick, class_names, rotation=45, ha="right"); plt.yticks(tick, class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    st.pyplot(fig)
    st.json(report)

@st.cache_resource
def load_any_model(path):
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "saved_model.pb")):
        return keras.models.load_model(path)
    elif os.path.isfile(path) and path.endswith(".h5"):
        return keras.models.load_model(path)
    else:
        raise ValueError(f"Path model tidak dikenal: {path}")

def preprocess_image_for_model(img: Image.Image, img_size=(224,224)):
    img = img.convert("RGB").resize(img_size)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)
    
def get_model_input_size(model):
    """Ambil ukuran input (H,W) dari model Keras; fallback ke (224,224)."""
    try:
        shp = model.input_shape  # (None, H, W, C)
        if isinstance(shp, list):  # untuk model multi-input
            shp = shp[0]
        h, w = int(shp[1]), int(shp[2])
        return (w, h) if w and h else (224, 224)
    except Exception:
        return (224, 224)

def load_labels_for_model(model_path, default_from_dataset=True, dataset_root="dataset"):
    """Cari labels.json di sebelah model; jika tak ada dan diizinkan, baca dari dataset/train_set."""
    # coba labels.json di sebelah file .keras/.h5
    base_dir = os.path.dirname(model_path)
    labels_json = os.path.join(base_dir, "labels.json")
    if os.path.isfile(labels_json):
        try:
            with open(labels_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # fallback: baca dari dataset/train_set (jika diinginkan)
    if default_from_dataset:
        train_dir = os.path.join(dataset_root, "train_set")
        if os.path.isdir(train_dir):
            return sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    return None


# ================== HALAMAN: ANALISIS ==================
if mode == "🧪 Analisis":
    st.header("🧪 Mode Analisis Sidik Jari")

    # kontrol khusus analisis di sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Pengaturan Analisis")
    dpi = st.sidebar.number_input("DPI (opsional)", 100, 2000, 500, step=50)
    block = st.sidebar.slider("Ukuran blok orientasi (px)", 8, 64, 24, step=4)
    thr = st.sidebar.slider("Ambang segmentasi (0–1)", 0.1, 0.9, 0.35, step=0.01)
    min_obj = st.sidebar.slider("Minimum area objek (px)", 32, 4000, 600, step=20)
    n_rays = st.sidebar.slider("Jumlah sinar radial", 8, 120, 36, step=2)
    max_radius = st.sidebar.slider("Radius maksimum (px)", 50, 1000, 300, step=10)
    show_vectors = st.sidebar.checkbox("Tampilkan bidang orientasi (preview)", value=False)
    show_minutiae = st.sidebar.checkbox("Tampilkan titik minutiae (estimasi)", value=True)
    use_manual_core = st.sidebar.checkbox("Gunakan core manual (draggable)", value=False)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Klasifikasi (opsional)")
    do_classify = st.sidebar.checkbox("Jalankan klasifikasi dengan model tersimpan", value=True)

    selected_model_path = None
    inf_model = None
    class_names_for_model = None
    model_input_size = (224, 224)

    if do_classify:
        candidates = list_saved_models("models")
        if not candidates:
            st.sidebar.warning("Belum ada model di folder `models/`.")
        else:
            selected_model_path = st.sidebar.selectbox("Pilih model", candidates, index=len(candidates)-1)
            if selected_model_path:
                try:
                    inf_model = load_any_model(selected_model_path)  # cached
                    class_names_for_model = load_labels_for_model(selected_model_path, default_from_dataset=True, dataset_root="dataset")
                    # tentukan ukuran input model
                    model_input_size = get_model_input_size(inf_model)
                    st.sidebar.caption(f"Input model: {model_input_size[0]}×{model_input_size[1]}")
                except Exception as e:
                    st.sidebar.error("Gagal memuat model untuk klasifikasi.")
                    st.sidebar.exception(e)
                    do_classify = False


    uploaded = st.file_uploader("Unggah gambar sidik jari (JPG/PNG/BMP)", type=["jpg","jpeg","png","bmp"])
    if uploaded is None:
        st.info("Silakan unggah gambar untuk memulai.")
        st.stop()

    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Gagal membaca gambar.")
        st.stop()

    gray = to_gray(bgr)
    norm = normalize(gray)

    theta, coherence = estimate_orientation(norm, block)
    fm = focus_measure(norm)
    enh_in, mask = segment_fingerprint(norm, thr=thr, min_area=min_obj)
    enh = enhance_ridges(enh_in)
    masked = enh.copy(); masked[~mask] = 0
    _, bw = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    skel = skeletonize((bw > 0) & mask); skel = remove_small_objects(skel, min_size=40)
    ends, bifs = detect_minutiae(skel)
    
    # ======= KLASIFIKASI (opsional di mode Analisis) =======
    if do_classify and inf_model is not None:
        try:
            # pakai gambar asli (bgr -> PIL RGB), resize sesuai input model
            pil_for_pred = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            arr = preprocess_image_for_model(pil_for_pred, img_size=(model_input_size[0], model_input_size[1]))
            preds = inf_model.predict(arr, verbose=0)[0]  # shape: (num_classes,)
            top_idx = int(np.argmax(preds))
            top_prob = float(preds[top_idx])

            # tentukan label
            if class_names_for_model and top_idx < len(class_names_for_model):
                pred_label = class_names_for_model[top_idx]
            else:
                pred_label = f"Class #{top_idx}"

            st.markdown("## 🤖 Hasil Klasifikasi (Model Terpilih)")
            st.write(f"**Prediksi:** {pred_label}")
            st.write(f"**Probabilitas:** {top_prob:.4f}")

            # tampilkan seluruh probabilitas
            if class_names_for_model:
                st.markdown("**Probabilitas per kelas:**")
                for i, p in enumerate(preds):
                    nm = class_names_for_model[i] if i < len(class_names_for_model) else f"Class #{i}"
                    st.write(f"- {nm}: {float(p):.4f}")
            else:
                st.code(json.dumps({str(i): float(p) for i, p in enumerate(preds)}, indent=2))
        except Exception as e:
            st.error("Klasifikasi gagal dijalankan.")
            st.exception(e)


    # deteksi core/delta otomatis
    core_cands = core_delta_candidates(theta, coherence, block, mask, want_positive=True, topk=2)
    delta_cands = core_delta_candidates(theta, coherence, block, mask, want_positive=False, topk=3)
    auto_core = core_cands[0][1] if core_cands else None
    auto_core_pi = core_cands[0][2] if core_cands else 0.0

    # kanvas core manual (dengan pre-seed circle)
    manual_core_xy = None
    if use_manual_core:
        st.subheader("🎯 Pilih / Geser Titik Core (Manual)")
        enh_rgb = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)
        bg_pil = Image.fromarray(enh_rgb)
        initial = None
        default_radius = 28
        if auto_core is not None:
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
        drawing_mode = "transform" if initial is not None else "circle"
        canvas = st_canvas(
            fill_color="rgba(255,255,0,0.25)",
            stroke_width=3,
            stroke_color="#ffff00",
            background_color="#00000000",
            background_image=bg_pil,
            update_streamlit=True,
            height=enh_rgb.shape[0],
            width=enh_rgb.shape[1],
            drawing_mode=drawing_mode,
            display_toolbar=True,
            key="core_canvas",
            initial_drawing=initial
        )
        if drawing_mode == "circle":
            st.info("Klik ikon **Circle** di toolbar, buat lingkaran, lalu geser ke pusat pola.")
        if canvas and canvas.json_data is not None:
            objs = canvas.json_data.get("objects", [])
            circle_obj = next((o for o in reversed(objs) if o.get("type")=="circle"), None)
            if circle_obj is not None:
                left = float(circle_obj.get("left", 0.0))
                top  = float(circle_obj.get("top", 0.0))
                radius = float(circle_obj.get("radius", 0.0))
                manual_core_xy = (int(round(left + radius)), int(round(top + radius)))
        if manual_core_xy is not None:
            st.success(f"Core manual: (x, y) = {manual_core_xy[0]}, {manual_core_xy[1]}")

    # pilih core final
    if use_manual_core and (manual_core_xy is not None):
        core, core_pi = manual_core_xy, 0.0
    else:
        core, core_pi = auto_core, auto_core_pi

    deltas = [(c[1], c[2]) for c in delta_cands]

    # ridge counts
    cd_counts, cd_lines = [], []
    if core is not None and deltas:
        for (dpt, dval) in deltas:
            cnt, (rr, cc) = count_crossings_on_line(skel, core, dpt)
            cd_counts.append({"delta_xy": (int(dpt[0]), int(dpt[1])), "poincare": float(dval), "ridge_count": int(cnt)})
            cd_lines.append((rr, cc))

    rad_counts, rays_pts = (None, [])
    if core is not None:
        rad_counts, rays_pts = count_crossings_from_center(skel.astype(np.uint8), core, n_rays=n_rays, max_radius=max_radius)

    # klasifikasi pola (heuristik)
    core_pts = [core] if core is not None else []
    delta_pts = [d[0] for d in deltas]
    pattern_label = classify_pattern(core_pts, delta_pts, skel.shape)

    # metrik
    roi_area_px = int(mask.sum())
    px_per_mm = (dpi / 25.4) if dpi else 500/25.4
    area_mm2 = roi_area_px / (px_per_mm**2 + 1e-8)
    coh_mean = float(np.mean(coherence[mask])) if mask.any() else 0.0

    # layout
    c1, c2, c3 = st.columns(3)
    c1.subheader("Asli"); c1.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
    c2.subheader("Enhancement"); c2.image(enh, clamp=True, use_column_width=True)
    viz = overlay_all(enh, core, deltas, cd_lines, rays_pts, ends, bifs, draw_minutiae=show_minutiae)
    c3.subheader("Skeleton + Overlay"); c3.image(viz, channels="RGB", use_column_width=True,
        caption="Kuning: Core • Merah: Delta • Oranye: Core→Delta • Hijau: sinar radial • Hijau tua: endings • Merah tua: bifurcations")

    st.markdown("### 📊 Ringkasan & Informasi")
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
        st.write(f"**Jumlah Delta:** {len(deltas)} (Poincaré ≈ −π).")
        for i, (pt, val) in enumerate(deltas, 1):
            dist = float(np.hypot(pt[0]-core[0], pt[1]-core[1])) if core is not None else 0.0
            st.write(f"- Delta #{i} @ ({int(pt[0])},{int(pt[1])}) | Poincaré: {val:.2f} | Jarak ke core: {dist:.1f}px")
    if cd_counts:
        st.markdown("#### Ridge Count – Core → Delta")
        for i, d in enumerate(cd_counts, 1):
            st.write(f"{i}. Delta @ {d['delta_xy']} | Poincaré: {d['poincare']:.2f} | **Ridge Count: {d['ridge_count']}**")
    if rad_counts is not None and len(rad_counts):
        st.markdown("#### Ridge Count – Radial dari Core")
        rA, rB, rC = st.columns(3)
        rA.metric("Rata-rata", f"{float(np.mean(rad_counts)):.2f}")
        rB.metric("Median", f"{float(np.median(rad_counts)):.2f}")
        rC.metric("Maksimum", f"{int(np.max(rad_counts))}")
        with st.expander("Detail per-arah (array)"):
            st.code(str(rad_counts.tolist()))

    if show_vectors:
        st.subheader("Bidang Orientasi (preview)")
        step = block
        Y, X = np.mgrid[step//2:gray.shape[0]:step, step//2:gray.shape[1]:step]
        U = np.cos(theta[step//2::step, step//2::step]); V = np.sin(theta[step//2::step, step//2::step])
        fig = plt.figure(); plt.imshow(gray, cmap='gray'); plt.quiver(X, Y, U, -V, color='r', pivot='mid', scale=30)
        if core is not None: plt.scatter([core[0]], [core[1]], c='y', s=40)
        if deltas:
            for (pt, _v) in deltas: plt.scatter([pt[0]], [pt[1]], c='r', s=30)
        st.pyplot(fig)

# ================== HALAMAN: TRAINING ==================
elif mode == "🏋️ Training (MobileNetV2)":
    st.header("🏋️ Training MobileNetV2 untuk Klasifikasi Sidik Jari")
    data_root = st.text_input("Folder dataset", value="dataset", help="Harus berisi train_set, val_set, test_set.")
    img_size = st.selectbox("Ukuran gambar (input model)", [(224,224), (192,192), (160,160)], index=0)
    colA, colB, colC = st.columns(3)
    with colA: batch_size = st.number_input("Batch size", 8, 128, 32, step=8)
    with colB: epochs = st.number_input("Epochs (tahap 1, freeze)", 1, 100, 10, step=1)
    with colC: lr = st.number_input("Learning rate", 1e-6, 1e-1, 1e-3, step=1e-4, format="%.6f")
    fine_tune = st.checkbox("Fine-tune backbone (unfreeze)", value=True)
    ft_epochs = st.number_input("Epochs (tahap 2, unfreeze)", 0, 100, 5, step=1)
    if st.button("Mulai Training"):
        with st.spinner("Menyiapkan data & melatih model..."):
            try:
                artifacts = train_mobilenet(
                    data_root=data_root,
                    img_size=img_size,
                    batch_size=int(batch_size),
                    epochs=int(epochs),
                    lr=float(lr),
                    fine_tune=bool(fine_tune and ft_epochs>0),
                    fine_tune_epochs=int(ft_epochs),
                    models_dir="models",
                )

                # tampilkan hasil training secara robust
                show_training_artifacts(artifacts)

                # evaluasi pada test set (opsional)
                st.subheader("Evaluasi pada Test Set")
                _, _, test_ds, class_names = build_datasets(
                    data_root=data_root, img_size=img_size, batch_size=int(batch_size)
                )
                evaluate_on_test(artifacts["model"], test_ds, class_names)
                # --- tampilkan lokasi file model & info (kompatibel .h5/.keras) ---
                best_path   = artifacts.get("best_model") or artifacts.get("best_h5")
                final_path  = artifacts.get("final_model") or artifacts.get("final_h5")
                labels_path = artifacts.get("labels_json")
                #saved_dir   = artifacts.get("saved_model_dir")  # mungkin None jika tidak export

                if best_path:
                    st.write("**Best model:**", best_path)
                if final_path:
                    st.write("**Final model:**", final_path)
                if labels_path:
                    st.write("**Labels JSON:**", labels_path)
                #if saved_dir:
                    #st.write("**SavedModel (export):**")

                st.write("**Kelas:**", artifacts["class_names"])
                st.write("**Test metrics:**", artifacts["test_metrics"])



            except Exception as e:
                st.error("Terjadi error saat training. Cek struktur dataset & log di bawah.")
                st.exception(e)
                st.stop()
        st.success("Training selesai! Model terbaik disimpan.")
        st.write("**Best H5:**", artifacts["best_h5"])
        #st.write("**SavedModel dir:**", artifacts["saved_model_dir"])
        st.write("**Kelas:**", artifacts["class_names"])
        st.write("**Test metrics:**", artifacts["test_metrics"])
        plot_history(artifacts["hist"], title="Tahap 1 (Freeze)")
        if artifacts["ft_hist"] is not None:
            plot_history(artifacts["ft_hist"], title="Tahap 2 (Fine-tune)")
        st.subheader("Evaluasi pada Test Set")
        _, _, test_ds, class_names = build_datasets(data_root=data_root, img_size=img_size, batch_size=int(batch_size))
        evaluate_on_test(artifacts["model"], test_ds, class_names)

# ================== HALAMAN: KLASIFIKASI ==================
elif mode == "🔎 Klasifikasi":
    st.header("🔎 Klasifikasi Sidik Jari dengan Model Tersimpan")
    models = list_saved_models("models")
    if not models:
        st.warning("Belum ada model di `models/`. Latih model dulu di menu Training.")
    else:
        selected = st.selectbox("Pilih model", models, index=len(models)-1)
        with st.spinner("Memuat model..."):
            try:
                inf_model = load_any_model(selected)
            except Exception as e:
                st.error("Gagal memuat model.")
                st.exception(e)
                st.stop()
        st.success("Model siap digunakan.")
        # baca class names dari dataset (jika ada)
        class_names = None
        train_dir = os.path.join("dataset", "train_set")
        if os.path.isdir(train_dir):
            class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        uploaded_img = st.file_uploader("Unggah gambar fingerprint", type=["jpg","jpeg","png","bmp"])
        if uploaded_img is not None:
            pil = Image.open(uploaded_img)
            st.image(pil, caption="Gambar diunggah", use_column_width=True)
            arr = preprocess_image_for_model(pil, img_size=(224,224))
            preds = inf_model.predict(arr, verbose=0)[0]
            top_idx = int(np.argmax(preds)); top_prob = float(preds[top_idx])
            label = class_names[top_idx] if class_names and top_idx < len(class_names) else f"Class #{top_idx}"
            st.subheader("Hasil Prediksi")
            st.write(f"**Prediksi:** {label}")
            st.write(f"**Probabilitas:** {top_prob:.4f}")
            st.markdown("**Probabilitas per kelas:**")
            if class_names:
                for i, p in enumerate(preds):
                    st.write(f"- {class_names[i]}: {float(p):.4f}")
            else:
                st.code(json.dumps({str(i): float(p) for i, p in enumerate(preds)}, indent=2))

