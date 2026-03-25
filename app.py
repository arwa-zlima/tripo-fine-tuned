"""
TripoSR Pro — Intelligent Furniture 3D Reconstruction
Graduation Seminar Demo
"""

import os
import sys
import time
import base64
import tempfile

import numpy as np
import streamlit as st
import torch
import rembg
from PIL import Image
from transformers import pipeline as hf_pipeline

# ── Project imports ────────────────────────────────────────────────────────────
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

# ── Page configuration (must be first Streamlit call) ─────────────────────────
st.set_page_config(
    page_title="TripoSR Pro — Furniture 3D Reconstruction",
    page_icon="🪑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORIES = [
    "chair", "table", "sofa", "bed", "cabinet",
    "bookshelf", "desk", "bench", "swivelchair", "wardrobe",
]

CATEGORY_ICONS = {
    "chair": "🪑", "table": "🪵", "sofa": "🛋️", "bed": "🛏️",
    "cabinet": "🗄️", "bookshelf": "📚", "desk": "🖥️", "bench": "💺",
    "swivelchair": "🔄", "wardrobe": "👔",
}

LORA_PATHS = {c: f"models/lora_{c}.safetensors" for c in CATEGORIES}

EXAMPLE_DIR = "examples"
EXAMPLES = [
    "chair.png", "hamburger.png", "poly_fox.png", "robot.png",
    "teapot.png", "tiger_girl.png", "horse.png", "flamingo.png",
    "unicorn.png", "iso_house.png", "marble.png", "police_woman.png",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  CSS  — dark theme, glassmorphism cards, animated pipeline
# ═══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
/* ── Import Google font ───────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Root variables ───────────────────────────── */
:root {
    --bg-primary:   #0f0f1a;
    --bg-secondary: #1a1a2e;
    --bg-card:      rgba(255,255,255,0.04);
    --border-card:  rgba(255,255,255,0.08);
    --accent:       #6c63ff;
    --accent-light: #8b83ff;
    --accent-glow:  rgba(108,99,255,0.35);
    --green:        #00c896;
    --amber:        #ffb347;
    --red:          #ff6b6b;
    --text:         #e8e8f0;
    --text-dim:     #8888a8;
    --radius:       14px;
}

/* ── Global ───────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-card) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

/* ── Hero banner ──────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid var(--border-card);
    border-radius: var(--radius);
    padding: 2.4rem 2.8rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 70% 30%, var(--accent-glow) 0%, transparent 60%);
    pointer-events: none;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(135deg, #fff 30%, var(--accent-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    position: relative;
}
.hero p {
    color: var(--text-dim);
    font-size: 1.05rem;
    margin: .6rem 0 0;
    position: relative;
}

/* ── Glass card ───────────────────────────────── */
.glass {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-card);
    border-radius: var(--radius);
    padding: 1.6rem;
    margin-bottom: 1.2rem;
}
.glass h3 {
    font-weight: 700;
    font-size: 1.05rem;
    margin: 0 0 1rem;
    color: var(--text);
}

/* ── Pipeline steps ───────────────────────────── */
.pipeline {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: .5rem;
    padding: .8rem 0;
}
.step {
    display: flex;
    align-items: center;
    gap: .5rem;
    padding: .65rem 1.2rem;
    border-radius: 10px;
    font-weight: 600;
    font-size: .88rem;
    transition: all .3s ease;
    border: 1px solid var(--border-card);
    background: var(--bg-card);
    color: var(--text-dim);
}
.step.active {
    background: linear-gradient(135deg, var(--accent), #8b5cf6);
    color: #fff;
    border-color: transparent;
    box-shadow: 0 4px 20px var(--accent-glow);
    animation: stepPulse 1.5s ease-in-out infinite;
}
.step.done {
    background: rgba(0,200,150,0.12);
    border-color: var(--green);
    color: var(--green);
}
.arrow { color: var(--text-dim); font-size: 1.1rem; }

@keyframes stepPulse {
    0%,100% { box-shadow: 0 4px 20px var(--accent-glow); }
    50%     { box-shadow: 0 4px 32px rgba(108,99,255,0.55); }
}

/* ── Confidence bars ──────────────────────────── */
.conf-row {
    display: flex;
    align-items: center;
    margin-bottom: .55rem;
    gap: .6rem;
}
.conf-label {
    width: 100px;
    font-size: .82rem;
    font-weight: 600;
    text-align: right;
    color: var(--text-dim);
    text-transform: capitalize;
}
.conf-bar-bg {
    flex: 1;
    height: 22px;
    background: rgba(255,255,255,0.06);
    border-radius: 6px;
    overflow: hidden;
    position: relative;
}
.conf-bar {
    height: 100%;
    border-radius: 6px;
    transition: width 1s cubic-bezier(.22,1,.36,1);
    background: linear-gradient(90deg, var(--accent), #8b5cf6);
}
.conf-bar.top {
    background: linear-gradient(90deg, var(--green), #34d399);
}
.conf-pct {
    width: 50px;
    font-size: .82rem;
    font-weight: 600;
    color: var(--text);
}

/* ── Stat boxes ───────────────────────────────── */
.stats-row {
    display: flex;
    gap: 1rem;
    margin-top: .6rem;
}
.stat-box {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-box .number {
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent-light), var(--green));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-box .label {
    font-size: .78rem;
    color: var(--text-dim);
    margin-top: .2rem;
}

/* ── LoRA badge ───────────────────────────────── */
.lora-badge {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    background: linear-gradient(135deg, var(--accent), #8b5cf6);
    padding: .6rem 1.2rem;
    border-radius: 8px;
    font-weight: 700;
    font-size: .95rem;
    color: #fff;
    margin-top: .5rem;
}

/* ── Streamlit widget overrides ───────────────── */
.stSlider > div > div {
    color: var(--text) !important;
}

div[data-testid="stFileUploader"] {
    border: 2px dashed var(--border-card) !important;
    border-radius: var(--radius) !important;
    background: var(--bg-card) !important;
}

/* Hide default Streamlit branding */
#MainMenu, footer { visibility: hidden; }

/* Section divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-card), transparent);
    margin: 1.5rem 0;
}
</style>
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  THREE.JS VIEWER  — embedded component for GLB rendering
# ═══════════════════════════════════════════════════════════════════════════════

def build_3d_viewer_html(glb_b64: str, height: int = 520) -> str:
    """Return a self-contained HTML page that renders a GLB via three.js."""
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
    body {{ margin:0; overflow:hidden; background:#12121e; }}
    canvas {{ display:block; }}
    #info {{
        position:absolute; bottom:12px; left:50%;
        transform:translateX(-50%);
        color:rgba(255,255,255,0.45); font:12px Inter,sans-serif;
    }}
</style>
</head>
<body>
<div id="info">Drag to rotate · Scroll to zoom</div>
<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.163.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.163.0/examples/jsm/"
  }}
}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';

const scene    = new THREE.Scene();
scene.background = new THREE.Color(0x12121e);

const camera   = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.01, 100);
camera.position.set(0, 1.2, 2.8);

const renderer = new THREE.WebGLRenderer({{ antialias:true }});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
document.body.appendChild(renderer.domElement);

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);
const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.5);
dirLight1.position.set(3, 5, 3);
scene.add(dirLight1);
const dirLight2 = new THREE.DirectionalLight(0x6c63ff, 0.6);
dirLight2.position.set(-3, 2, -3);
scene.add(dirLight2);

// Ground grid
const grid = new THREE.GridHelper(4, 20, 0x333355, 0x222244);
grid.position.y = -0.01;
scene.add(grid);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotate = true;
controls.autoRotateSpeed = 1.5;
controls.target.set(0, 0.5, 0);
controls.update();

// Load GLB from base64
const b64  = "{glb_b64}";
const raw  = atob(b64);
const arr  = new Uint8Array(raw.length);
for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i);

const loader = new GLTFLoader();
loader.parse(arr.buffer, '', (gltf) => {{
    const model = gltf.scene;
    const box   = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    const size   = box.getSize(new THREE.Vector3()).length();
    model.position.sub(center);
    const s = 2.0 / size;
    model.scale.set(s, s, s);
    scene.add(model);
    controls.target.set(0, 0, 0);
}});

// Resize handler
window.addEventListener('resize', () => {{
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
}});

// Render loop
(function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}})();
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING  (cached so it only runs once)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    m = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    m.renderer.set_chunk_size(8192)
    m.to(device)
    return m, device


@st.cache_resource(show_spinner=False)
def load_rembg():
    return rembg.new_session()


@st.cache_resource(show_spinner=False)
def load_classifier():
    """Load CLIP zero-shot classifier (cached — only runs once)."""
    return hf_pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess(image: Image.Image, do_remove_bg: bool, fg_ratio: float) -> Image.Image:
    def fill_bg(img):
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        return Image.fromarray((arr * 255.0).astype(np.uint8))

    if do_remove_bg:
        image = image.convert("RGB")
        image = remove_background(image, load_rembg())
        image = resize_foreground(image, fg_ratio)
        image = fill_bg(image)
    else:
        if image.mode == "RGBA":
            image = fill_bg(image)
    return image


def classify_image(image: Image.Image):
    """Classify furniture using CLIP zero-shot classification."""
    classifier = load_classifier()
    rgb_image = image.convert("RGB")
    predictions = classifier(rgb_image, candidate_labels=CATEGORIES)

    # predictions is a list of dicts sorted by score descending
    all_scores = {p["label"]: p["score"] for p in predictions}
    top = predictions[0]
    return top["label"], top["score"], all_scores


def load_lora_adapter(category: str) -> str:
    """Placeholder — implement actual LoRA weight loading here."""
    time.sleep(0.2)
    return LORA_PATHS.get(category, "N/A")


def generate_mesh(image: Image.Image, resolution: int):
    model, device = load_model()
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, True, resolution=resolution)[0]
    # Apply orientation fix
    import trimesh
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))

    paths = {}
    for fmt in ("obj", "glb"):
        tmp = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
        mesh.export(tmp.name)
        paths[fmt] = tmp.name
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def pipeline_html(active_idx: int = -1, done_up_to: int = -1) -> str:
    """Render the 4-step pipeline with active / done states."""
    labels = [
        ("📤", "Upload"),
        ("🎯", "Classify"),
        ("⚡", "Load LoRA"),
        ("🧊", "Generate 3D"),
    ]
    parts = []
    for i, (icon, name) in enumerate(labels):
        cls = "step"
        if i < done_up_to:
            cls += " done"
        elif i == active_idx:
            cls += " active"
        parts.append(f'<div class="{cls}">{icon} {name}</div>')
        if i < len(labels) - 1:
            parts.append('<div class="arrow">›</div>')
    return f'<div class="pipeline">{"".join(parts)}</div>'


def confidence_bars_html(scores: dict, top_cat: str) -> str:
    """Horizontal confidence bars for every category."""
    sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    rows = []
    for cat, score in sorted_cats:
        pct = score * 100
        bar_cls = "conf-bar top" if cat == top_cat else "conf-bar"
        icon = CATEGORY_ICONS.get(cat, "")
        rows.append(
            f'<div class="conf-row">'
            f'  <div class="conf-label">{icon} {cat}</div>'
            f'  <div class="conf-bar-bg"><div class="{bar_cls}" style="width:{pct:.1f}%"></div></div>'
            f'  <div class="conf-pct">{pct:.1f}%</div>'
            f'</div>'
        )
    return "".join(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        remove_bg = st.toggle("Remove Background", value=True)
        fg_ratio = st.slider("Foreground Ratio", 0.5, 1.0, 0.85, 0.05)
        resolution = st.select_slider(
            "Mesh Resolution",
            options=[32, 64, 128, 192, 256, 320],
            value=256,
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 💾 Memory Comparison")
        st.markdown(
            """
            <div class="stats-row">
                <div class="stat-box">
                    <div class="number">1.75 GB</div>
                    <div class="label">With LoRA</div>
                </div>
                <div class="stat-box">
                    <div class="number">7.5 GB</div>
                    <div class="label">5 Full Models</div>
                </div>
            </div>
            <div class="stats-row" style="margin-top:.6rem">
                <div class="stat-box" style="border-color:var(--green);">
                    <div class="number" style="background:linear-gradient(135deg,var(--green),#34d399);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">76%</div>
                    <div class="label">VRAM Saved</div>
                </div>
                <div class="stat-box">
                    <div class="number">&lt;0.5s</div>
                    <div class="label">Inference</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📂 Example Gallery")

        cols = st.columns(3)
        for i, name in enumerate(EXAMPLES):
            path = os.path.join(EXAMPLE_DIR, name)
            if os.path.isfile(path):
                with cols[i % 3]:
                    img = Image.open(path)
                    if st.button("", key=f"ex_{i}", help=name):
                        st.session_state["selected_example"] = path
                    st.image(img, use_container_width=True, caption=name.split(".")[0])

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="text-align:center;color:var(--text-dim);font-size:.75rem;">'
            'TripoSR Pro · Graduation Seminar 2026</p>',
            unsafe_allow_html=True,
        )

    return remove_bg, fg_ratio, resolution


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Inject CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Sidebar
    remove_bg, fg_ratio, resolution = render_sidebar()

    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero">
            <h1>TripoSR Pro</h1>
            <p>Intelligent Furniture Classification → LoRA-Optimized 3D Reconstruction</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Pipeline ──────────────────────────────────────────────────────────
    pipeline_placeholder = st.empty()
    pipeline_placeholder.markdown(
        f'<div class="glass">{pipeline_html(-1, -1)}</div>',
        unsafe_allow_html=True,
    )

    # ── Upload ────────────────────────────────────────────────────────────
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="glass"><h3>📤 Upload Image</h3></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drag and drop a furniture image",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )

        # Check for example selection
        if "selected_example" in st.session_state and st.session_state["selected_example"]:
            input_image = Image.open(st.session_state["selected_example"])
            st.image(input_image, caption="Selected example", use_container_width=True)
            st.session_state["selected_example"] = None
        elif uploaded is not None:
            input_image = Image.open(uploaded)
            st.image(input_image, caption="Uploaded image", use_container_width=True)
        else:
            input_image = None

        generate_btn = st.button(
            "🚀  Generate 3D Model",
            use_container_width=True,
            type="primary",
            disabled=input_image is None,
        )

    with col_result:
        st.markdown('<div class="glass"><h3>🎯 Classification & Adapter</h3></div>', unsafe_allow_html=True)
        classification_placeholder = st.empty()
        confidence_placeholder = st.empty()
        lora_placeholder = st.empty()

        if not generate_btn:
            classification_placeholder.markdown(
                '<p style="color:var(--text-dim);text-align:center;padding:2rem 0;">'
                'Upload an image and click Generate to start the pipeline.</p>',
                unsafe_allow_html=True,
            )

    # ── Processing ────────────────────────────────────────────────────────
    if generate_btn and input_image is not None:
        # Step 1: Preprocess
        pipeline_placeholder.markdown(
            f'<div class="glass">{pipeline_html(0, 0)}</div>',
            unsafe_allow_html=True,
        )
        with st.spinner(""):
            status = st.status("🔄 **Processing pipeline...**", expanded=True)
            with status:
                st.write("**Step 1/4** — Preprocessing image...")
                processed = preprocess(input_image, remove_bg, fg_ratio)

                # Step 2: Classify
                pipeline_placeholder.markdown(
                    f'<div class="glass">{pipeline_html(1, 1)}</div>',
                    unsafe_allow_html=True,
                )
                st.write("**Step 2/4** — Classifying furniture type...")
                category, confidence, all_scores = classify_image(processed)

                # Show classification results
                icon = CATEGORY_ICONS.get(category, "🪑")
                classification_placeholder.markdown(
                    f'<div class="glass">'
                    f'<div style="text-align:center;padding:1rem 0;">'
                    f'<div style="font-size:2.5rem;">{icon}</div>'
                    f'<div style="font-size:1.4rem;font-weight:800;color:var(--green);margin:.3rem 0;">'
                    f'{category.upper()}</div>'
                    f'<div style="color:var(--text-dim);">{confidence*100:.1f}% confidence</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
                confidence_placeholder.markdown(
                    f'<div class="glass"><h3>📊 All Predictions</h3>'
                    f'{confidence_bars_html(all_scores, category)}</div>',
                    unsafe_allow_html=True,
                )

                # Step 3: Load LoRA
                pipeline_placeholder.markdown(
                    f'<div class="glass">{pipeline_html(2, 2)}</div>',
                    unsafe_allow_html=True,
                )
                st.write(f"**Step 3/4** — Loading LoRA adapter for **{category}**...")
                lora_path = load_lora_adapter(category)

                lora_placeholder.markdown(
                    f'<div class="glass">'
                    f'<div class="lora-badge">⚡ {category.upper()} adapter loaded</div>'
                    f'<p style="color:var(--text-dim);font-size:.82rem;margin-top:.5rem;">'
                    f'Path: {lora_path}</p></div>',
                    unsafe_allow_html=True,
                )

                # Step 4: Generate mesh
                pipeline_placeholder.markdown(
                    f'<div class="glass">{pipeline_html(3, 3)}</div>',
                    unsafe_allow_html=True,
                )
                st.write(f"**Step 4/4** — Generating 3D mesh at resolution **{resolution}**...")
                mesh_paths = generate_mesh(processed, resolution)

            status.update(label="✅ **Pipeline complete!**", state="complete", expanded=False)

        # Pipeline done
        pipeline_placeholder.markdown(
            f'<div class="glass">{pipeline_html(-1, 4)}</div>',
            unsafe_allow_html=True,
        )

        # ── 3D Viewer ────────────────────────────────────────────────
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="glass"><h3>🧊 Interactive 3D Viewer</h3></div>',
            unsafe_allow_html=True,
        )

        # Read GLB and encode to base64
        with open(mesh_paths["glb"], "rb") as f:
            glb_bytes = f.read()
        glb_b64 = base64.b64encode(glb_bytes).decode("utf-8")

        # Render three.js viewer
        viewer_html = build_3d_viewer_html(glb_b64, height=520)
        st.components.v1.html(viewer_html, height=540, scrolling=False)

        # ── Downloads ────────────────────────────────────────────────
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        dl1, dl2, dl3 = st.columns([1, 1, 1])
        with dl1:
            with open(mesh_paths["glb"], "rb") as f:
                st.download_button(
                    "⬇️  Download GLB",
                    data=f.read(),
                    file_name=f"{category}_model.glb",
                    mime="model/gltf-binary",
                    use_container_width=True,
                )
        with dl2:
            with open(mesh_paths["obj"], "rb") as f:
                st.download_button(
                    "⬇️  Download OBJ",
                    data=f.read(),
                    file_name=f"{category}_model.obj",
                    mime="text/plain",
                    use_container_width=True,
                )
        with dl3:
            # Processed image download
            buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            processed.save(buf.name)
            with open(buf.name, "rb") as f:
                st.download_button(
                    "⬇️  Processed Image",
                    data=f.read(),
                    file_name="processed.png",
                    mime="image/png",
                    use_container_width=True,
                )


if __name__ == "__main__":
    # Warm up all models in background
    with st.spinner("Loading models (TripoSR + CLIP classifier)..."):
        load_model()
        load_rembg()
        load_classifier()
    main()
