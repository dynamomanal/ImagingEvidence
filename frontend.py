import io
import os
import sys
import tempfile
import warnings
import logging
import streamlit as st

# Ensure stdout/stderr can handle Unicode on Windows (cp1252 console default)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)


# ══════════════════════════════════════════════════════════════
#  MODALITY CONFIG
# ══════════════════════════════════════════════════════════════

MODALITIES = [
    "Echo",
    "Cardiac MRI",
    "Cardiac CT",
]

MODALITY_ICONS = {
    "Echo":        "🫀",
    "Cardiac MRI": "🧲",
    "Cardiac CT":  "💠",
}

MODALITY_VIEWS = {
    "Echo": [
        "Not specified",
        "Parasternal Long Axis (PLAX)",
        "Parasternal Short Axis (PSAX)",
        "Apical 4-Chamber (A4C)",
        "Apical 2-Chamber (A2C)",
        "Apical 3-Chamber (A3C)",
        "Apical 5-Chamber (A5C)",
        "Subcostal 4-Chamber",
        "Subcostal IVC",
        "Suprasternal",
        "Tissue Doppler (TDI)",
        "Colour Doppler",
        "Spectral Doppler (PW/CW)",
        "M-Mode",
        "Other",
    ],
    "Cardiac MRI": [
        "Not specified",
        "Cine 4-Chamber",
        "Cine 2-Chamber",
        "Cine 3-Chamber (LVOT)",
        "Cine Short Axis",
        "Late Gadolinium Enhancement (LGE)",
        "T1 Mapping",
        "T2 Mapping / STIR",
        "Phase Contrast Flow",
        "First-Pass Perfusion",
        "Aortic / Great Vessels",
        "Right Heart",
        "Other",
    ],
    "Cardiac CT": [
        "Not specified",
        "Axial — Coronary",
        "Coronal MPR",
        "Sagittal MPR",
        "Curved MPR — LAD",
        "Curved MPR — LCX",
        "Curved MPR — RCA",
        "3D Volume Rendering",
        "Calcium Score",
        "Pericardium",
        "Pulmonary Veins",
        "Aorta / TAVR Planning",
        "Other",
    ],
}

MODALITY_HINTS = {
    "Echo":        "JPG PNG GIF MP4 AVI · DICOM (.dcm)",
    "Cardiac MRI": "DICOM (.dcm) · JPG PNG · multi-frame cine supported",
    "Cardiac CT":  "DICOM (.dcm) · JPG PNG · CT windowing applied automatically",
}

EMPTY_STATE_CHIPS = {
    "Echo": [
        "Integrated multi-view report", "LV function &amp; EF",
        "Valvular disease", "Doppler interpretation", "Differential diagnosis",
    ],
    "Cardiac MRI": [
        "LGE scar pattern", "Volumes &amp; EF", "T1 / T2 tissue mapping",
        "Myocarditis vs ischemia", "Cardiomyopathy workup",
    ],
    "Cardiac CT": [
        "Coronary stenosis grading", "Calcium score", "Plaque morphology",
        "Aortic dimensions", "Incidental findings",
    ],
}


# ══════════════════════════════════════════════════════════════
#  GIF HELPERS
# ══════════════════════════════════════════════════════════════

def _gif_info(raw: bytes) -> int:
    try:
        img = Image.open(io.BytesIO(raw))
        return getattr(img, "n_frames", 1)
    except Exception:
        return 1


def extract_gif_frames(raw: bytes, n: int) -> list:
    gif    = Image.open(io.BytesIO(raw))
    total  = getattr(gif, "n_frames", 1)
    n      = min(n, total)
    frames = []
    for i in range(n):
        idx = int(i * total / n)
        try:
            gif.seek(idx)
            frames.append(gif.convert("RGB").copy())
        except EOFError:
            break
    return frames


# ══════════════════════════════════════════════════════════════
#  VIDEO HELPERS (MP4 / AVI / MOV)
# ══════════════════════════════════════════════════════════════

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}

def _is_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in VIDEO_EXTS


def extract_video_frames(raw: bytes, filename: str, n: int) -> tuple:
    import cv2
    suffix   = Path(filename).suffix.lower() or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        cap   = cv2.VideoCapture(tmp_path)
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        n     = min(n, total)
        frames = []
        for i in range(n):
            idx = int(i * total / n)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        cap.release()
        return frames, total
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════
#  DICOM HELPERS
# ══════════════════════════════════════════════════════════════

def _is_dicom(filename: str) -> bool:
    return Path(filename).suffix.lower() in {".dcm", ".dicom", ""}


def _dicom_frame_count(raw: bytes) -> int:
    """Return number of frames in a DICOM file (1 if single-frame)."""
    try:
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(raw))
        n  = getattr(ds, "NumberOfFrames", 1)
        return int(n) if n else 1
    except Exception:
        return 1


def _apply_windowing(arr, ds):
    """Apply DICOM window center/width to a numpy array, then normalize to uint8."""
    import numpy as np
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth",  None)
    if wc is not None and ww is not None:
        if hasattr(wc, "__iter__"):
            wc, ww = float(wc[0]), float(ww[0])
        else:
            wc, ww = float(wc), float(ww)
        arr = arr.copy()
        arr = arr.astype(float)
        lo, hi = wc - ww / 2, wc + ww / 2
        arr = arr.clip(lo, hi)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn) * 255.0
    return arr.astype("uint8")


def _dicom_to_pil(raw: bytes, frame_idx: int = 0) -> tuple:
    """
    Convert one frame of a DICOM file to a PIL RGB Image.
    Applies Hounsfield rescale + window/level from the DICOM header.
    Returns (PIL.Image, metadata dict).
    """
    import pydicom
    import numpy as np

    ds  = pydicom.dcmread(io.BytesIO(raw))
    arr = ds.pixel_array.astype(float)

    # Rescale slope / intercept (Hounsfield units for CT)
    slope     = float(getattr(ds, "RescaleSlope",     1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept

    # Select frame for multi-frame DICOM
    if arr.ndim == 3:
        frame_idx = min(frame_idx, arr.shape[0] - 1)
        arr = arr[frame_idx]

    arr = _apply_windowing(arr, ds)
    img = Image.fromarray(arr).convert("RGB")

    meta = {
        "modality":    str(getattr(ds, "Modality",          "Unknown")),
        "rows":        int(getattr(ds, "Rows",               0)),
        "cols":        int(getattr(ds, "Columns",            0)),
        "n_frames":    int(getattr(ds, "NumberOfFrames",     1) or 1),
        "series_desc": str(getattr(ds, "SeriesDescription",  "")),
        "pixel_sp":    getattr(ds, "PixelSpacing",           None),
    }
    return img, meta


def extract_dicom_frames(raw: bytes, n: int) -> tuple:
    """
    Extract n evenly-spaced frames from a multi-frame DICOM file.
    Returns (frames: list[PIL.Image], total: int, meta: dict).
    """
    import pydicom
    import numpy as np

    ds  = pydicom.dcmread(io.BytesIO(raw))
    arr = ds.pixel_array.astype(float)

    slope     = float(getattr(ds, "RescaleSlope",     1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept

    if arr.ndim == 3:
        total = arr.shape[0]
    else:
        total = 1
        arr   = arr[np.newaxis]

    # Window all frames together for consistent brightness
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth",  None)
    if wc is not None and ww is not None:
        if hasattr(wc, "__iter__"):
            wc, ww = float(wc[0]), float(ww[0])
        else:
            wc, ww = float(wc), float(ww)
        arr = arr.clip(wc - ww / 2, wc + ww / 2)

    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn) * 255.0
    arr = arr.astype("uint8")

    n = min(n, total)
    frames = []
    for i in range(n):
        idx = int(i * total / n)
        frames.append(Image.fromarray(arr[idx]).convert("RGB"))

    meta = {
        "modality":    str(getattr(ds, "Modality",         "Unknown")),
        "series_desc": str(getattr(ds, "SeriesDescription", "")),
        "n_frames":    total,
    }
    return frames, total, meta


# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG & DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Imaging Evidence",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --sb:     #0F172A;
    --sb2:    #1E293B;
    --sb3:    #334155;
    --sb-bd:  #1E293B;
    --sb-t:   #94A3B8;
    --sb-t2:  #CBD5E1;
    --sb-dim: #475569;

    --bg:     #0F172A;
    --s:      #1E293B;
    --ink:    #FFFFFF;
    --ink2:   #E2E8F0;
    --mu:     #94A3B8;
    --mu2:    #64748B;
    --bd:     #1E293B;
    --bd2:    #334155;

    --ac:     #0B3D2E;
    --ac2:    #1A5C41;
    --acl:    #F0FDF4;
    --acm:    #DCFCE7;

    --r:  10px;
    --rs: 6px;
    --rp: 999px;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--bg) !important;
    color: var(--ink) !important;
    -webkit-font-smoothing: antialiased !important;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

.block-container {
    max-width: 820px !important;
    padding: 0 1.8rem 2rem !important;
    padding-top: 1.4rem !important;
    margin: 0 auto !important;
}

section[data-testid="stSidebar"] {
    background: var(--sb) !important;
    border-right: 1px solid var(--sb-bd) !important;
    min-width: 268px !important;
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem !important; }

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] small { color: var(--sb-t) !important; }
section[data-testid="stSidebar"] hr {
    border-color: var(--sb-bd) !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] img {
    border-radius: var(--rs);
    border: 1px solid var(--sb2);
    opacity: 0.92;
}

section[data-testid="stSidebar"] div[data-testid="stFileUploader"] section {
    background: var(--sb2) !important;
    border: 1.5px dashed var(--sb-dim) !important;
    border-radius: var(--rs) !important;
    transition: border-color .15s ease;
}
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] section:hover {
    border-color: var(--sb-t) !important;
}
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button {
    background: var(--sb3) !important;
    color: var(--sb-t2) !important;
    border: 1px solid var(--sb-dim) !important;
    border-radius: var(--rs) !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: var(--sb2) !important;
    border-color: var(--sb-dim) !important;
    color: var(--sb-t2) !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: var(--sb-t2) !important;
    background: var(--sb2) !important;
}

section[data-testid="stSidebar"] div[data-testid="stExpander"] {
    background: var(--sb2) !important;
    border: 1px solid var(--sb-bd) !important;
    border-radius: var(--rs) !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary { color: var(--sb-t) !important; }
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary:hover { color: var(--sb-t2) !important; }

section[data-testid="stSidebar"] .stButton > button {
    background: var(--sb2) !important;
    color: var(--sb-t2) !important;
    border: 1px solid var(--sb-bd) !important;
    border-radius: var(--rs) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.005em !important;
    transition: all .12s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--sb3) !important;
    color: #F1F5F9 !important;
    border-color: var(--sb-dim) !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: var(--ac) !important;
    color: #FFFFFF !important;
    border-color: var(--ac) !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: var(--ac2) !important;
    border-color: var(--ac2) !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:disabled {
    background: var(--sb2) !important;
    color: var(--sb-dim) !important;
    border-color: var(--sb-bd) !important;
}
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .stCaption * { color: var(--sb-dim) !important; }

div[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 1.6rem 0 !important;
    margin-bottom: 0 !important;
    box-shadow: none !important;
    border-bottom: 1px solid var(--bd2) !important;
}
div[data-testid="stChatMessage"]:last-of-type { border-bottom: none !important; }

div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--bd2) !important;
    border: 1px solid var(--bd) !important;
    border-radius: var(--r) !important;
    padding: 1rem 1.15rem !important;
    margin-bottom: 0.15rem !important;
}

div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li { color: var(--ink); font-size: 0.955rem; line-height: 1.74; }

div[data-testid="stChatMessage"] h1,
div[data-testid="stChatMessage"] h2 {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    color: var(--mu) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin: 1.6rem 0 0.55rem !important;
    padding-bottom: 0.4rem !important;
    border-bottom: 1px solid var(--bd2) !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-testid="stChatMessage"] h3 {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--mu2) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin: 1.2rem 0 0.4rem !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-testid="stChatMessage"] h4 {
    font-size: 0.93rem !important;
    font-weight: 600 !important;
    color: var(--ink2) !important;
    margin: 0.85rem 0 0.3rem !important;
    letter-spacing: -0.01em !important;
}
div[data-testid="stChatMessage"] strong { color: var(--ink2); font-weight: 600; }
div[data-testid="stChatMessage"] code {
    background: var(--acl) !important;
    color: var(--ac) !important;
    padding: 0.12em 0.42em;
    border-radius: 4px;
    font-size: 0.87em;
}
div[data-testid="stChatMessage"] blockquote {
    border-left: 3px solid var(--ac) !important;
    padding-left: 0.9rem !important;
    color: var(--mu) !important;
    margin: 0.75rem 0 !important;
    font-style: italic;
}
div[data-testid="stChatMessage"] table {
    border-collapse: collapse; width: 100%; font-size: 0.88rem; margin: 0.9rem 0;
}
div[data-testid="stChatMessage"] th {
    background: var(--bd2) !important;
    color: var(--mu) !important;
    font-weight: 600; font-size: 0.73rem; text-transform: uppercase;
    letter-spacing: 0.07em; padding: 0.5rem 0.8rem;
    border-bottom: 2px solid var(--bd); text-align: left;
}
div[data-testid="stChatMessage"] td {
    padding: 0.5rem 0.8rem; border-bottom: 1px solid var(--bd2); color: var(--ink);
}
div[data-testid="stChatMessage"] tr:last-child td { border-bottom: none; }
div[data-testid="stChatMessageContainer"] { background: transparent !important; }

div[data-testid="stChatInput"] {
    background: var(--s) !important;
    border: 1.5px solid var(--bd) !important;
    border-radius: var(--r) !important;
    box-shadow: 0 1px 3px rgba(15,23,42,.05), 0 4px 14px rgba(15,23,42,.04) !important;
    transition: border-color .15s ease, box-shadow .15s ease;
}
div[data-testid="stChatInput"]:focus-within {
    border-color: var(--ac) !important;
    box-shadow: 0 0 0 3px rgba(11,61,46,.07) !important;
}
div[data-testid="stChatInput"] textarea {
    color: var(--ink) !important;
    font-size: 0.95rem !important;
    font-family: 'Inter', sans-serif !important;
}

div[data-testid="stAlert"] {
    background: var(--acl) !important;
    border: 1px solid var(--acm) !important;
    border-left: 3px solid var(--ac) !important;
    border-radius: var(--rs) !important;
    color: var(--ink) !important;
}
div[data-testid="stSpinner"] p { color: var(--mu) !important; font-size: 0.87rem !important; }
div[data-testid="stChatMessage"] img { border-radius: var(--rs); border: 1px solid var(--bd); }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bd); border-radius: var(--rp); }
::-webkit-scrollbar-thumb:hover { background: var(--mu2); }
hr { border-color: var(--bd2) !important; margin: 0.75rem 0 !important; }

/* always-open sidebar — hide collapse/expand toggle */
button[data-testid="collapsedControl"],
button[data-testid="baseButton-headerNoPadding"],
button[data-testid="stSidebarNavCollapseButton"],
button[data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* force sidebar visible even if browser localStorage has it collapsed */
section[data-testid="stSidebar"] {
    transform: none !important;
    margin-left: 0 !important;
    visibility: visible !important;
}
</style>
""", unsafe_allow_html=True)


# Force sidebar open on every load (clears any browser-stored collapsed state)
import streamlit.components.v1 as _components
_components.html("""
<script>
(function() {
    // Clear any Streamlit sidebar collapsed state from localStorage
    for (var key in localStorage) {
        if (key.toLowerCase().includes('sidebar')) {
            localStorage.removeItem(key);
        }
    }
    // Click the expand button if sidebar is collapsed
    function tryOpen() {
        var btn = document.querySelector('[data-testid="stSidebarCollapsedControl"]') ||
                  document.querySelector('[data-testid="collapsedControl"]');
        if (btn) btn.click();
    }
    setTimeout(tryOpen, 300);
    setTimeout(tryOpen, 800);
})();
</script>
""", height=0)


# ── session state ─────────────────────────────────────────────────────────────
_defaults = {
    "messages":         [],
    "images":           [],
    "img_names":        [],
    "view_labels":      [],
    "modality":         "Echo",
    "history":          [],
    "last_findings":    None,
    "last_literature":  None,
    "last_modality":    "Echo",
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
def _sb_label(text):
    st.markdown(
        f"<div style='font-size:.68rem;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:.09em;color:#475569;margin:.9rem 0 .4rem;'>{text}</div>",
        unsafe_allow_html=True,
    )

with st.sidebar:

    # ── brand ──
    st.markdown(
        "<div style='padding-bottom:1.1rem;margin-bottom:.6rem;"
        "border-bottom:1px solid #1E293B;'>"
        "<div style='font-size:1.15rem;font-weight:700;color:#E2E8F0;"
        "letter-spacing:-.025em;'>Imaging<span style='color:#34D399'>.</span>Evidence</div>"
        "<div style='font-size:.72rem;color:#475569;margin-top:.15rem;'>"
        "Multimodal cardiac imaging AI</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── new study ──
    if st.button("+ New study", use_container_width=True):
        for _k in ("messages","images","img_names","view_labels","last_findings","last_literature"):
            st.session_state[_k] = [] if isinstance(st.session_state[_k], list) else None
        st.rerun()

    # ── modality selector ──
    _sb_label("Imaging modality")
    selected_modality = st.selectbox(
        "modality",
        options=MODALITIES,
        index=MODALITIES.index(st.session_state.modality),
        format_func=lambda m: f"{MODALITY_ICONS.get(m,'')}  {m}",
        key="modality_select",
        label_visibility="collapsed",
    )
    if selected_modality != st.session_state.modality:
        st.session_state.modality    = selected_modality
        st.session_state.images      = []
        st.session_state.img_names   = []
        st.session_state.view_labels = []
        st.rerun()

    modality     = st.session_state.modality
    view_options = MODALITY_VIEWS.get(modality, ["Not specified", "Other"])

    # ── upload ──
    _sb_label(f"{MODALITY_ICONS.get(modality,'')} {modality} images")

    uploaded_files = st.file_uploader(
        "upload",
        type=["jpg", "jpeg", "png", "tiff", "gif",
              "mp4", "avi", "mov", "webm",
              "dcm", "dicom"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help=MODALITY_HINTS.get(modality, "JPG PNG DICOM"),
    )

    if uploaded_files:
        st.session_state.images    = []
        st.session_state.img_names = []
        all_labels: list           = []

        n_up = len(uploaded_files)
        st.markdown(
            f"<div style='font-size:.78rem;color:#34D399;font-weight:500;"
            f"margin:.45rem 0 .4rem;'>&#10003; {n_up} file{'s' if n_up>1 else ''} loaded</div>",
            unsafe_allow_html=True,
        )

        for i, f in enumerate(uploaded_files):
            raw      = f.read()
            fname    = f.name.lower()
            is_gif   = fname.endswith(".gif")
            is_video = _is_video(f.name)
            is_dcm   = fname.endswith(".dcm") or fname.endswith(".dicom")

            if is_gif:
                n_total     = _gif_info(raw)
                is_animated = n_total > 1
            else:
                n_total = 0
                is_animated = False

            # ── expander title ──
            if is_animated:
                title_suffix = f" · GIF · {n_total}f"
            elif is_video:
                title_suffix = f" · {Path(f.name).suffix.upper()[1:]}"
            elif is_dcm:
                title_suffix = " · DICOM"
            else:
                title_suffix = ""

            with st.expander(f"#{i+1} · {f.name[:24]}{title_suffix}", expanded=False):

                # ── DICOM ─────────────────────────────────────────────────
                if is_dcm:
                    try:
                        dcm_frames = _dicom_frame_count(raw)
                        if dcm_frames > 1:
                            # Multi-frame DICOM (cine MRI, multi-slice CT)
                            st.caption(f"Multi-frame DICOM · {dcm_frames} frames")
                            n_extract = st.slider(
                                "Frames to extract",
                                min_value=1, max_value=min(dcm_frames, 12),
                                value=min(4, dcm_frames), key=f"dcm_nf_{i}",
                                help="Evenly sampled across the series.",
                            )
                            with st.spinner("Reading DICOM frames..."):
                                frames, total_f, meta = extract_dicom_frames(raw, n_extract)
                            # thumbnail strip
                            thumb_cols = st.columns(len(frames))
                            for fi, (col, frame) in enumerate(zip(thumb_cols, frames)):
                                with col:
                                    st.image(frame, width="stretch", caption=f"f{fi+1}")
                            for fi, frame in enumerate(frames):
                                st.session_state.images.append(frame)
                                st.session_state.img_names.append(
                                    f"{f.name} [frame {fi+1}/{len(frames)}]"
                                )
                            n_added = len(frames)
                            desc    = meta.get("series_desc", "")
                            dicom_m = meta.get("modality", "")
                            st.caption(
                                f"{len(raw)/1024:.1f} KB · {total_f} frames · "
                                f"{n_extract} extracted"
                                + (f" · {dicom_m}" if dicom_m else "")
                                + (f" · {desc}" if desc else "")
                            )
                        else:
                            # Single-frame DICOM
                            with st.spinner("Reading DICOM..."):
                                img, meta = _dicom_to_pil(raw, frame_idx=0)
                            st.image(img, width="stretch")
                            st.session_state.images.append(img)
                            st.session_state.img_names.append(f.name)
                            n_added = 1
                            desc    = meta.get("series_desc", "")
                            dicom_m = meta.get("modality", "")
                            st.caption(
                                f"{len(raw)/1024:.1f} KB · "
                                f"{meta['rows']}x{meta['cols']}px"
                                + (f" · {dicom_m}" if dicom_m else "")
                                + (f" · {desc}" if desc else "")
                            )
                    except Exception as e:
                        st.error(f"Could not read DICOM: {e}")
                        n_added = 0

                # ── ANIMATED GIF ──────────────────────────────────────────
                elif is_animated:
                    st.image(raw, width="stretch",
                             caption=f"Animated GIF · {n_total} frames")
                    n_extract = st.slider(
                        "Frames to extract",
                        min_value=1, max_value=min(n_total, 8),
                        value=min(4, n_total), key=f"gif_nf_{i}",
                        help="Sampled evenly across the cardiac cycle.",
                    )
                    frames = extract_gif_frames(raw, n_extract)
                    for fi, frame in enumerate(frames):
                        st.session_state.images.append(frame)
                        st.session_state.img_names.append(
                            f"{f.name} [frame {fi+1}/{n_extract}]"
                        )
                    n_added = n_extract
                    st.caption(
                        f"{len(raw)/1024:.1f} KB · {n_total} frames · {n_extract} extracted"
                    )

                # ── VIDEO ─────────────────────────────────────────────────
                elif is_video:
                    st.video(raw)
                    n_extract = st.slider(
                        "Frames to extract",
                        min_value=1, max_value=8,
                        value=4, key=f"vid_nf_{i}",
                        help="Evenly sampled from the full video duration.",
                    )
                    with st.spinner("Extracting frames..."):
                        try:
                            frames, n_total_vid = extract_video_frames(raw, f.name, n_extract)
                        except Exception as e:
                            st.error(f"Could not read video: {e}")
                            frames, n_total_vid = [], 0
                    if frames:
                        thumb_cols = st.columns(len(frames))
                        for fi, (col, frame) in enumerate(zip(thumb_cols, frames)):
                            with col:
                                st.image(frame, width="stretch", caption=f"f{fi+1}")
                        for fi, frame in enumerate(frames):
                            st.session_state.images.append(frame)
                            st.session_state.img_names.append(
                                f"{f.name} [frame {fi+1}/{len(frames)}]"
                            )
                    n_added = len(frames)
                    st.caption(
                        f"{len(raw)/1024:.1f} KB · {n_total_vid} total frames · {n_added} extracted"
                    )

                # ── STATIC IMAGE ──────────────────────────────────────────
                else:
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    st.session_state.images.append(img)
                    st.session_state.img_names.append(f.name)
                    st.image(img, width="stretch")
                    n_added = 1
                    st.caption(f"{len(raw)/1024:.1f} KB · {img.size[0]}x{img.size[1]}px")

                # ── view / series label ──
                label = st.selectbox(
                    "view / series",
                    view_options,
                    key=f"vl_{i}",
                    label_visibility="collapsed",
                )
                all_labels.extend([label] * n_added)

        st.session_state.view_labels = all_labels

    # ── analysis flow (Groq only) ──
    st.session_state["flow_selection"] = "Flow A - Groq Vision  (cloud · instant)"
    _sb_label("Analysis engine")
    st.markdown(
        "<div style='background:#0B3D2E;border:1px solid #1A5C41;border-radius:6px;"
        "padding:.45rem .75rem;font-size:.82rem;color:#34D399;font-weight:500;'>"
        "&#9711; Groq Vision &nbsp;·&nbsp; cloud · instant</div>",
        unsafe_allow_html=True,
    )
    st.caption("Groq Vision API — no local GPU needed. Supports all modalities.")

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    analyze_clicked = st.button(
        f"Analyze {MODALITY_ICONS.get(modality,'')}  ->",
        use_container_width=True,
        type="primary",
        disabled=len(st.session_state.images) == 0,
    )

    # ── session history ──
    if st.session_state.history:
        _sb_label("This session")
        for entry in reversed(st.session_state.history[-6:]):
            _icon = MODALITY_ICONS.get(entry.get("modality", "Echo"), "🩻")
            st.markdown(
                f"<div style='padding:.38rem .55rem;border-radius:5px;font-size:.8rem;"
                f"color:#94A3B8;background:#1E293B;margin-bottom:.22rem;"
                f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>"
                f"{_icon} {entry['label']}</div>",
                unsafe_allow_html=True,
            )

    # ── info expander ──
    st.markdown("---")
    with st.expander("Supported modalities", expanded=False):
        for m in MODALITIES:
            st.markdown(f"**{MODALITY_ICONS.get(m,'')} {m}**  \n{MODALITY_HINTS.get(m,'')}\n")

    _badge_color, _badge_text = "#34D399", "Groq Cloud · instant"

    st.markdown(
        f"<div style='font-size:.68rem;color:{_badge_color};text-align:center;"
        f"margin-top:.7rem;font-weight:500;'>&#9711; {_badge_text}</div>"
        "<div style='font-size:.65rem;color:#94A3B8;text-align:center;"
        "line-height:1.75;margin-top:.4rem;'>"
        "Clinician review required · Not a diagnostic device"
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
#  TOPBAR
# ══════════════════════════════════════════════════════════════
n_views  = len(st.session_state.images)
modality = st.session_state.modality

_status = ""
if n_views > 0:
    _label  = f"{n_views} image{'s' if n_views > 1 else ''} ready"
    _status = (
        f"<span style='display:inline-flex;align-items:center;gap:.35rem;"
        f"background:#0B3D2E;border:1px solid #1A5C41;color:#34D399;"
        f"font-size:.72rem;font-weight:500;padding:.22rem .65rem;border-radius:999px;'>"
        f"<span style='width:6px;height:6px;background:#10B981;"
        f"border-radius:50%;display:inline-block;'></span>"
        f"{_label}</span>"
    )

_flow_label = "Groq Vision"

st.markdown(
    f"<div style='display:flex;align-items:center;justify-content:space-between;"
    f"padding:.8rem 0 1rem 0;border-bottom:1px solid #1E293B;margin-bottom:0;'>"
    f"<div style='font-size:1rem;font-weight:700;color:#34D399;letter-spacing:-.02em;'>"
    f"Imaging<span style='color:#10B981'>.</span>Evidence"
    f"<span style='font-size:.72rem;color:#475569;font-weight:400;margin-left:.6rem;'>"
    f"{MODALITY_ICONS.get(modality,'')} {modality}</span></div>"
    f"<div style='display:flex;align-items:center;gap:.5rem;'>{_status}"
    f"<span style='background:#1E293B;border:1px solid #334155;color:#94A3B8;"
    f"font-size:.7rem;font-weight:500;padding:.2rem .6rem;border-radius:999px;'>"
    f"{_flow_label}"
    f"</span></div></div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════
#  CHAT THREAD / EMPTY STATE
# ══════════════════════════════════════════════════════════════
if not st.session_state.messages:
    chips = EMPTY_STATE_CHIPS.get(modality, EMPTY_STATE_CHIPS["Echo"])
    st.markdown(
        "<div style='text-align:center;padding:4.5rem 1rem 3rem;'>"
        f"<div style='font-size:2rem;opacity:.2;margin-bottom:1.2rem;'>{MODALITY_ICONS.get(modality,'🩻')}</div>"
        "<div style='font-size:1.45rem;font-weight:600;color:#FFFFFF;"
        "letter-spacing:-.025em;margin-bottom:.55rem;'>"
        f"Upload a {modality} study</div>"
        "<div style='font-size:.9rem;color:#94A3B8;max-width:380px;"
        "margin:0 auto 2.5rem;line-height:1.65;'>"
        f"Add one or more {modality} images from the left panel, "
        "then run the analysis to receive a guideline-referenced cardiac report."
        "</div>"
        "<div style='display:flex;flex-wrap:wrap;gap:.45rem;"
        "justify-content:center;max-width:520px;margin:0 auto;'>",
        unsafe_allow_html=True,
    )
    for chip in chips:
        st.markdown(
            f"<div style='background:#1E293B;border:1px solid #334155;color:#E2E8F0;"
            f"padding:.4rem .85rem;border-radius:999px;font-size:.82rem;'>{chip}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div></div>", unsafe_allow_html=True)

else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "images" in msg and msg["images"]:
                imgs = msg["images"]
                cols = st.columns(min(len(imgs), 4))
                for idx, (im, nm) in enumerate(zip(imgs, msg.get("img_names", []))):
                    with cols[idx % 4]:
                        st.image(im, caption=nm, width="stretch")

            if msg.get("is_compare"):
                _nv  = msg.get("n_views", "?")
                _mod = msg.get("modality", "Cardiac")
                st.markdown(
                    f"### {_mod} Report - {_nv} image{'s' if _nv != 1 else ''} - Compare Mode"
                )
                st.markdown(
                    "<div style='font-size:.78rem;color:#94A3B8;margin:.15rem 0 1rem;'>"
                    "Stage 1 descriptions - Flow A (Groq Vision) vs Flow B (MedGemma)"
                    "</div>",
                    unsafe_allow_html=True,
                )
                _col_a, _col_b = st.columns(2, gap="medium")
                with _col_a:
                    st.markdown(
                        "<div style='background:#0B3D2E;border:1px solid #1A5C41;"
                        "border-radius:8px;padding:.55rem .85rem .4rem;margin-bottom:.55rem;'>"
                        "<span style='font-size:.68rem;font-weight:700;color:#34D399;"
                        "text-transform:uppercase;letter-spacing:.09em;'>"
                        "Flow A - Groq Vision</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(msg.get("findings_a", "_No output_"))
                with _col_b:
                    _b_err = msg.get("findings_b_error", False)
                    if _b_err:
                        st.markdown(
                            "<div style='background:#451A03;border:1px solid #92400E;"
                            "border-radius:8px;padding:.55rem .85rem .4rem;margin-bottom:.55rem;'>"
                            "<span style='font-size:.68rem;font-weight:700;color:#FCD34D;"
                            "text-transform:uppercase;letter-spacing:.09em;'>"
                            "Flow B - MedGemma &#9888;</span></div>",
                            unsafe_allow_html=True,
                        )
                        st.warning(msg.get("findings_b", "_No output_"))
                    else:
                        st.markdown(
                            "<div style='background:#1C1917;border:1px solid #44403C;"
                            "border-radius:8px;padding:.55rem .85rem .4rem;margin-bottom:.55rem;'>"
                            "<span style='font-size:.68rem;font-weight:700;color:#A8A29E;"
                            "text-transform:uppercase;letter-spacing:.09em;'>"
                            "Flow B - MedGemma</span></div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(msg.get("findings_b", "_No output_"))
                st.markdown("---")
                st.markdown(
                    "<div style='font-size:.68rem;font-weight:700;color:#94A3B8;"
                    "text-transform:uppercase;letter-spacing:.09em;margin:.5rem 0 .3rem;'>"
                    "Synthesis - Based on Flow A findings</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(msg["content"])
                _np = msg.get("n_papers", 0)
                st.markdown(f"*Sources: {_np} papers retrieved from PubMed · PubMed Central*")
            else:
                st.markdown(msg["content"])


# ══════════════════════════════════════════════════════════════
#  ANALYZE HANDLER
# ══════════════════════════════════════════════════════════════
if analyze_clicked and st.session_state.images:
    n        = len(st.session_state.images)
    modality = st.session_state.modality
    labels   = st.session_state.get("view_labels", ["Not specified"] * n)
    icon     = MODALITY_ICONS.get(modality, "")

    view_desc = ", ".join(f"Image {i+1} ({labels[i]})" for i in range(n))
    question  = (
        f"I am providing {n} {modality} image{'s' if n > 1 else ''} for analysis: {view_desc}. "
        f"Please analyze all images and provide a comprehensive {modality} report."
    )

    short = (labels[0] if labels and labels[0] != "Not specified" else modality)
    if n > 1:
        short += f" +{n - 1} more"
    st.session_state.history.append({"label": short, "n": n, "modality": modality})

    st.session_state.messages.append({
        "role":      "user",
        "content":   (
            f"**{n} {modality} image{'s' if n > 1 else ''} submitted for analysis.**\n\n"
            f"_{view_desc}_"
        ),
        "images":    list(st.session_state.images),
        "img_names": [f"{labels[i]} - {st.session_state.img_names[i]}" for i in range(n)],
    })

    try:
        from agent import run_groq_vision, run_literature_search, run_synthesis

        findings   = None
        literature = None
        synthesis  = None

        with st.status("Running 3-stage analysis pipeline...", expanded=True) as status:
            st.write(f"Stage 1 — Analysing {modality} with Groq Vision...")
            findings = run_groq_vision(st.session_state.images, question, modality)
            st.write(f"Stage 1 complete — {len(findings)} chars")

            st.write("Stage 2 — Searching PubMed & PubMed Central...")
            literature = run_literature_search(findings, modality)
            n_pub = len(literature.get("pubmed", []))
            n_pmc = len(literature.get("pmc", []))
            n_sch = len(literature.get("scholar", []))
            st.write(f"Stage 2 complete — {n_pub} PubMed + {n_pmc} PMC + {n_sch} Scholar")

            st.write("Stage 3 — Synthesizing with Groq / LLaMA-3...")
            synthesis = run_synthesis(findings, literature, modality)
            st.write("Stage 3 complete — report generated")
            status.update(label="Analysis complete", state="complete", expanded=False)

        n_papers = (
            len((literature or {}).get("pubmed",  [])) +
            len((literature or {}).get("pmc",     [])) +
            len((literature or {}).get("scholar", []))
        )
        st.session_state.last_findings   = findings
        st.session_state.last_literature = literature
        st.session_state.last_modality   = modality
        st.session_state.messages.append({
            "role":    "assistant",
            "content": (
                f"### {modality} Report — {n} image{'s' if n > 1 else ''} · Groq Vision\n\n"
                f"{synthesis}\n\n"
                f"---\n"
                f"*Sources: {n_papers} papers retrieved from PubMed · PubMed Central · Semantic Scholar*"
            ),
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        st.session_state.messages.append({
            "role":    "assistant",
            "content": f"**Analysis error:**\n```\n{tb}\n```",
        })
    st.rerun()


# ══════════════════════════════════════════════════════════════
#  FOLLOW-UP INPUT
# ══════════════════════════════════════════════════════════════
_last_findings = st.session_state.get("last_findings")
if _last_findings:
    from vector_store import get_store as _gs
    _store_count = _gs().count
    st.markdown(
        f"<div style='font-size:.75rem;color:#34D399;text-align:center;"
        f"margin:.4rem 0 .2rem;font-weight:500;'>"
        f"&#9711; RAG active &nbsp;·&nbsp; {_store_count} papers indexed"
        f"</div>",
        unsafe_allow_html=True,
    )

user_input = st.chat_input("Ask a follow-up question about this study...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    last_findings   = st.session_state.get("last_findings")
    last_literature = st.session_state.get("last_literature")
    last_modality   = st.session_state.get("last_modality", "Echo")

    if last_findings:
        with st.spinner("Searching literature and generating response..."):
            try:
                from agent import answer_followup
                result = answer_followup(
                    question=user_input,
                    findings=last_findings,
                    literature=last_literature,
                    history=st.session_state.messages[:-1],
                    modality=last_modality,
                )
                st.session_state.messages.append({"role": "assistant", "content": result})
            except Exception as e:
                import traceback
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": f"**Error answering follow-up:** `{e}`\n```\n{traceback.format_exc()}\n```",
                })
    elif st.session_state.images:
        st.session_state.messages.append({
            "role":    "assistant",
            "content": "Please click **Analyze** first to run the initial analysis.",
        })
    else:
        st.session_state.messages.append({
            "role":    "assistant",
            "content": "No study is loaded. Upload images from the left panel first.",
        })
    st.rerun()


# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown(
    "<div style='text-align:center;color:#94A3B8;font-size:.72rem;"
    "padding:1.5rem 0 .5rem;border-top:1px solid #1E293B;margin-top:2rem;line-height:1.9;'>"
    "Imaging Evidence &nbsp;·&nbsp; Research preview &nbsp;·&nbsp; "
    "Outputs require clinician verification before any clinical decision."
    "</div>",
    unsafe_allow_html=True,
)
