import io
import os
import sys
import tempfile
import warnings
import logging
import streamlit as st

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

MODALITIES = ["Echocardiogram", "MRI", "CT Scan"]

MODALITY_ICONS = {
    "Echocardiogram": "🫀",
    "MRI":            "🧲",
    "CT Scan":        "💠",
}

MODALITY_HINTS = {
    "Echocardiogram": "JPG · PNG · GIF · MP4 · AVI · DICOM",
    "MRI":            "DICOM · JPG · PNG",
    "CT Scan":        "DICOM · JPG · PNG",
}

EMPTY_STATE_CHIPS = {
    "Echocardiogram": [
        "Full cardiac report", "LV function & EF",
        "Valve assessment", "Doppler analysis", "Differential diagnosis",
    ],
    "MRI": [
        "LGE scar pattern", "Volumes & EF", "Tissue mapping",
        "Myocarditis vs ischemia", "Cardiomyopathy",
    ],
    "CT Scan": [
        "Coronary stenosis", "Calcium score", "Plaque morphology",
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
#  VIDEO HELPERS
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
    try:
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(raw))
        n  = getattr(ds, "NumberOfFrames", 1)
        return int(n) if n else 1
    except Exception:
        return 1


def _apply_windowing(arr, ds):
    import numpy as np
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth",  None)
    if wc is not None and ww is not None:
        if hasattr(wc, "__iter__"):
            wc, ww = float(wc[0]), float(ww[0])
        else:
            wc, ww = float(wc), float(ww)
        arr = arr.copy().astype(float)
        arr = arr.clip(wc - ww / 2, wc + ww / 2)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn) * 255.0
    return arr.astype("uint8")


def _dicom_to_pil(raw: bytes, frame_idx: int = 0) -> tuple:
    import pydicom, numpy as np
    ds  = pydicom.dcmread(io.BytesIO(raw))
    arr = ds.pixel_array.astype(float)
    slope     = float(getattr(ds, "RescaleSlope",     1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept
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
    import pydicom, numpy as np
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
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Imaging Evidence",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── DESIGN SYSTEM ─────────────────────────────────────────────
# Light mode  : white/light-blue body  +  dark-navy sidebar
# Dark mode   : near-black body        +  deep-navy sidebar
# Accent      : #1A56DB (blue) light  /  #3B82F6 (blue) dark
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── LIGHT MODE TOKENS (default) ──────────────────────── */
:root {
    /* body — pure white + navy text */
    --bg:      #FFFFFF;
    --s:       #F5F8FF;
    --s2:      #EEF2FF;
    --ink:     #0D1F52;
    --ink2:    #1A3A8F;
    --mu:      #2451B7;
    --mu2:     #6B8FBE;
    --bd:      #C7D9FF;
    --bd2:     #A8C4FF;

    /* sidebar */
    --sb:      #1E3A5F;
    --sb2:     #243F6A;
    --sb3:     #2E4F7E;
    --sb-bd:   #2E4F7E;
    --sb-t:    #BFDBFE;
    --sb-t2:   #E0EEFF;
    --sb-dim:  #7EAAD4;

    /* accent — blue */
    --ac:      #1A56DB;
    --ac2:     #1447C0;
    --acl:     #EBF2FF;
    --acm:     #BFDBFE;

    --r:  10px;
    --rs: 6px;
    --rp: 999px;
}

/* ── DARK MODE TOKENS (toggled via JS) ─────────────────── */
body.dark-mode {
    --bg:      #0A0F1E;
    --s:       #111827;
    --s2:      #1A2236;
    --ink:     #E8F1FF;
    --ink2:    #A8C4E8;
    --mu:      #6B8FBE;
    --mu2:     #4A6A96;
    --bd:      #1E3055;
    --bd2:     #243A66;
    --sb:      #060D1A;
    --sb2:     #0D1930;
    --sb3:     #142240;
    --sb-bd:   #142240;
    --sb-t:    #7EB3E8;
    --sb-t2:   #B8D4F0;
    --sb-dim:  #3D608A;
    --ac:      #3B82F6;
    --ac2:     #2563EB;
    --acl:     #0D1F3C;
    --acm:     #1E3A5F;
}

/* ── BASE ─────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--bg) !important;
    color: var(--ink) !important;
    -webkit-font-smoothing: antialiased !important;
}
/* Remove Streamlit's native header entirely — no space, no padding offset */
#MainMenu, footer,
header[data-testid="stHeader"],
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stAppViewBlockContainer"] { padding-top: 0 !important; }
[data-testid="stBottom"] { padding-bottom: 0 !important; }

.block-container {
    max-width: 820px !important;
    padding: 0 1.8rem 2rem !important;
    padding-top: 1rem !important;
    margin: 0 auto !important;
}

/* ── SIDEBAR ─────────────────────────────────────────── */
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

/* Uploader */
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

/* Selectbox */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: var(--sb2) !important;
    border-color: var(--sb-dim) !important;
    color: var(--sb-t2) !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: var(--sb-t2) !important;
    background: var(--sb2) !important;
}

/* Expander */
section[data-testid="stSidebar"] div[data-testid="stExpander"] {
    background: var(--sb2) !important;
    border: 1px solid var(--sb-bd) !important;
    border-radius: var(--rs) !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary { color: var(--sb-t) !important; }
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary:hover { color: var(--sb-t2) !important; }

/* Buttons */
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

/* ── CHAT MESSAGES ───────────────────────────────────── */
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
    background: var(--s2) !important;
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
    background: var(--s2) !important;
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

/* ── CHAT INPUT ──────────────────────────────────────── */
div[data-testid="stChatInput"] {
    background: var(--s) !important;
    border: 1.5px solid var(--bd) !important;
    border-radius: var(--r) !important;
    box-shadow: 0 1px 3px rgba(26,86,219,.05), 0 4px 14px rgba(26,86,219,.04) !important;
    transition: border-color .15s ease, box-shadow .15s ease;
}
div[data-testid="stChatInput"]:focus-within {
    border-color: var(--ac) !important;
    box-shadow: 0 0 0 3px rgba(26,86,219,.08) !important;
}
div[data-testid="stChatInput"] textarea {
    color: var(--ink) !important;
    font-size: 0.95rem !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── ALERTS / SPINNERS ──────────────────────────────── */
div[data-testid="stAlert"] {
    background: var(--acl) !important;
    border: 1px solid var(--acm) !important;
    border-left: 3px solid var(--ac) !important;
    border-radius: var(--rs) !important;
    color: var(--ink) !important;
}
div[data-testid="stSpinner"] p { color: var(--mu) !important; font-size: 0.87rem !important; }
div[data-testid="stChatMessage"] img { border-radius: var(--rs); border: 1px solid var(--bd); }

/* Prevent browser pulling viewport down as new content is appended */
[data-testid="stAppViewContainer"],
[data-testid="stVerticalBlock"] { overflow-anchor: none; }

/* ── SCROLLBAR ──────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bd2); border-radius: var(--rp); }
::-webkit-scrollbar-thumb:hover { background: var(--mu2); }
hr { border-color: var(--bd2) !important; margin: 0.75rem 0 !important; }

/* ── MOBILE ─────────────────────────────────────────── */
#sb-hamburger {
    display: none;
    background: none;
    border: none;
    cursor: pointer;
    padding: .4rem .5rem;
    margin-right: .4rem;
    color: var(--ink);
    flex-shrink: 0;
    line-height: 0;
    border-radius: 8px;
    transition: background .12s ease;
    min-width: 40px;
    min-height: 40px;
    align-items: center;
    justify-content: center;
    -webkit-tap-highlight-color: transparent;
    touch-action: manipulation;
}
#sb-hamburger:hover { background: rgba(26,86,219,.1); }
#sb-hamburger:active { background: rgba(26,86,219,.2); }

/* Sidebar backdrop — tapping outside closes sidebar on mobile */
#sb-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.45);
    z-index: 999;
    -webkit-tap-highlight-color: transparent;
}

@media (min-width: 769px) {
    button[data-testid="collapsedControl"],
    button[data-testid="baseButton-headerNoPadding"],
    button[data-testid="stSidebarNavCollapseButton"],
    button[data-testid="stSidebarCollapsedControl"] { display: none !important; }
    section[data-testid="stSidebar"] {
        transform: none !important;
        margin-left: 0 !important;
        visibility: visible !important;
    }
    #sb-overlay { display: none !important; }
}

@media (max-width: 768px) {
    /* Topbar: fixed at top of viewport, always visible */
    #ie-topbar {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 100 !important;
        background: var(--bg) !important;
        padding: 0.55rem 0.75rem 0.6rem !important;
        border-bottom: 1px solid var(--bd) !important;
        margin-bottom: 0 !important;
    }
    section[data-testid="stSidebar"] {
        min-width: 82vw !important;
        max-width: 85vw !important;
        z-index: 1000 !important;
        position: fixed !important;
        height: 100vh !important;
        top: 0 !important;
        left: 0 !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        transition: none !important;
    }
    .block-container {
        max-width: 100% !important;
        padding: 0 0.75rem 1.5rem !important;
        /* Reserve space for fixed topbar (~52px) */
        padding-top: 3.6rem !important;
    }
    div[data-testid="stChatMessage"] { padding: 0.85rem 0 !important; }
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li { font-size: 0.895rem; line-height: 1.65; }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        padding: 0.7rem 0.9rem !important;
    }
    div[data-testid="stChatInput"] { border-radius: var(--rs) !important; }
    #sb-hamburger { display: inline-flex !important; align-items: center; }
    button[data-testid="collapsedControl"],
    button[data-testid="stSidebarCollapsedControl"] {
        position: fixed !important;
        top: -9999px !important;
        left: -9999px !important;
        width: 1px !important;
        height: 1px !important;
        opacity: 0 !important;
        overflow: hidden !important;
    }
    .mobile-empty-state { padding-top: 2rem !important; padding-bottom: 1.5rem !important; }
}
/* ── MOBILE SIDEBAR COMPACT + STICKY ANALYZE ───────────────── */
@media (max-width: 768px) {
    /* Remove subtitle to save ~20px */
    #sb-brand-sub { display: none !important; }

    /* Compact brand block */
    #sb-brand {
        padding-bottom: .55rem !important;
        margin-bottom: .35rem !important;
    }

    /* Reduce sidebar inner top padding */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0.75rem !important;
    }

    /* ── Replace << with ✕ on the sidebar close button ── */
    section[data-testid="stSidebar"] button[data-testid="stSidebarNavCollapseButton"] {
        width: 34px !important;
        height: 34px !important;
        min-width: 34px !important;
        border-radius: 8px !important;
        background: rgba(255,255,255,.08) !important;
        border: 1px solid rgba(191,219,254,.18) !important;
        padding: 0 !important;
        cursor: pointer !important;
        position: relative !important;
        overflow: hidden !important;
        margin-bottom: .5rem !important;
    }
    /* hide every child (the << SVG) */
    section[data-testid="stSidebar"] button[data-testid="stSidebarNavCollapseButton"] * {
        display: none !important;
    }
    /* draw ✕ via pseudo-element — no DOM/event interference */
    section[data-testid="stSidebar"] button[data-testid="stSidebarNavCollapseButton"]::after {
        content: '\00D7' !important;
        display: block !important;
        font-size: 20px !important;
        font-weight: 300 !important;
        color: #BFDBFE !important;
        line-height: 1 !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
    }

    /* Sticky Analyze / Add-to-Study button — always visible at bottom */
    section[data-testid="stSidebar"] .stButton:has(button[kind="primary"]) {
        position: sticky !important;
        bottom: 0 !important;
        z-index: 10 !important;
        background: linear-gradient(
            to bottom,
            rgba(30,58,95,0) 0%,
            var(--sb) 32%
        ) !important;
        padding-bottom: .6rem !important;
        padding-top: .55rem !important;
        margin-top: .1rem !important;
    }
}

@media (max-width: 480px) {
    #ie-topbar {
        padding: 0.55rem 0.4rem 0.6rem !important;
    }
    .block-container {
        padding: 0 0.4rem 1rem !important;
        padding-top: 3.6rem !important;
    }
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li { font-size: 0.855rem; }
    div[data-testid="stChatInput"] textarea { font-size: 0.875rem !important; }
    #topbar-status span:last-child { display: none; }
}

/* Tables: horizontal scroll so they don't break layout on mobile */
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    max-width: 100%;
}

/* ── THEME TOGGLE BUTTON ────────────────────────────────── */
#theme-toggle {
    background: none;
    border: 1px solid var(--bd);
    cursor: pointer;
    padding: .32rem .42rem;
    border-radius: 8px;
    color: var(--ink);
    line-height: 0;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    transition: background .12s ease, border-color .12s ease;
    flex-shrink: 0;
}
#theme-toggle:hover { background: var(--s2); border-color: var(--bd2); }
#theme-toggle svg { display: block; }
</style>
""", unsafe_allow_html=True)


# ── Splash screen ─────────────────────────────────────────────
import streamlit.components.v1 as _components
_components.html("""
<script>
(function() {
    var par = window.parent || window;
    var pd  = par.document;

    if (par.sessionStorage.getItem('ie-splash-seen')) return;
    if (pd.getElementById('ie-splash')) return;

    var s = pd.createElement('style');
    s.textContent = [
        '#ie-splash{position:fixed;inset:0;z-index:999999;display:flex;align-items:center;',
        'justify-content:center;padding:1.5rem;',
        'background:radial-gradient(ellipse 80% 60% at 50% 40%,#0F2444 0%,#060D1A 100%);',
        'animation:ieSplashIn .5s cubic-bezier(.22,.68,0,1.2) both;}',

        '@keyframes ieSplashIn{from{opacity:0}to{opacity:1}}',
        '@keyframes ieSplashSlide{from{opacity:0;transform:translateY(28px)}to{opacity:1;transform:translateY(0)}}',
        '@keyframes ieSplashOut{from{opacity:1;transform:scale(1)}to{opacity:0;transform:scale(1.04)}}',
        '@keyframes ieGlow{0%,100%{opacity:.18}50%{opacity:.32}}',

        '#ie-splash-inner{text-align:center;max-width:560px;width:100%;',
        'animation:ieSplashSlide .7s cubic-bezier(.22,.68,0,1.2) .15s both;}',

        '#ie-splash-glow{position:absolute;top:50%;left:50%;transform:translate(-50%,-58%);',
        'width:min(680px,90vw);height:min(480px,60vw);border-radius:50%;',
        'background:radial-gradient(ellipse,rgba(59,130,246,.22) 0%,transparent 70%);',
        'animation:ieGlow 4s ease-in-out infinite;pointer-events:none;}',

        '.ie-badge{display:inline-flex;align-items:center;gap:.45rem;',
        'background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.3);',
        'color:#60A5FA;font-size:.7rem;font-weight:600;letter-spacing:.1em;',
        'text-transform:uppercase;padding:.3rem .85rem;border-radius:999px;margin-bottom:1.6rem;}',
        '.ie-badge-dot{width:6px;height:6px;background:#3B82F6;border-radius:50%;',
        'box-shadow:0 0 6px #3B82F6;animation:ieGlow 2s ease-in-out infinite;}',

        '.ie-brand{font-size:clamp(2rem,6vw,3rem);font-weight:800;color:#E8F1FF;',
        'letter-spacing:-.04em;line-height:1;margin-bottom:.5rem;}',
        '.ie-brand-dot{color:#3B82F6;}',

        '.ie-headline{font-size:clamp(.95rem,2.5vw,1.2rem);font-weight:500;',
        'color:#93C5FD;margin-bottom:1.4rem;line-height:1.5;}',

        '.ie-body{font-size:clamp(.82rem,2vw,.96rem);color:#7EAAD4;line-height:1.8;',
        'margin-bottom:.6rem;max-width:480px;margin-left:auto;margin-right:auto;}',

        '.ie-body strong{color:#BFDBFE;}',

        '.ie-body-accent{font-size:clamp(.82rem,2vw,.95rem);color:#93C5FD;font-weight:500;',
        'margin-bottom:2.2rem;font-style:italic;}',

        '.ie-btn{display:inline-flex;align-items:center;gap:.5rem;',
        'background:linear-gradient(135deg,#1A56DB,#3B82F6);color:#fff;',
        'border:none;border-radius:10px;padding:.8rem 2.2rem;',
        'font-size:1rem;font-weight:600;letter-spacing:-.01em;cursor:pointer;',
        'box-shadow:0 4px 24px rgba(59,130,246,.35);',
        'transition:transform .15s,box-shadow .15s;margin-bottom:1.4rem;}',
        '.ie-btn:hover{transform:translateY(-2px);box-shadow:0 8px 32px rgba(59,130,246,.45);}',
        '.ie-btn:active{transform:translateY(0);}',

        '.ie-disclaimer{font-size:.68rem;color:#3D608A;letter-spacing:.03em;}',

        '.ie-divider{width:48px;height:2px;',
        'background:linear-gradient(90deg,transparent,#1A56DB,transparent);',
        'margin:1.4rem auto;border-radius:999px;}',

        '.ie-pills{display:flex;flex-wrap:wrap;gap:.5rem;justify-content:center;margin-bottom:1.8rem;}',
        '.ie-pill{background:rgba(26,86,219,.15);border:1px solid rgba(59,130,246,.2);',
        'color:#60A5FA;font-size:.72rem;padding:.28rem .75rem;border-radius:999px;}',
    ].join('');
    pd.head.appendChild(s);

    var el = pd.createElement('div');
    el.id = 'ie-splash';
    el.innerHTML = [
        '<div id="ie-splash-glow"></div>',
        '<div id="ie-splash-inner">',

        '  <div class="ie-badge">',
        '    <span class="ie-badge-dot"></span>',
        '    Research Preview',
        '  </div>',

        '  <div class="ie-brand">',
        '    Imaging<span class="ie-brand-dot">.</span>Evidence',
        '  </div>',

        '  <div class="ie-headline">',
        '    The first AI built exclusively for cardiac imaging interpretation',
        '  </div>',

        '  <div class="ie-divider"></div>',

        '  <div class="ie-body">',
        '    Upload a <strong>DICOM, MRI, CT Scan or Echocardiogram</strong> study.',
        '    A Vision Language Model reads every frame, cross-references',
        '    thousands of peer-reviewed cardiology papers in real time via RAG,',
        '    and returns a fully structured clinical report — in seconds.',
        '  </div>',

        '  <div class="ie-body-accent">No tool like this has existed before.</div>',

        '  <div class="ie-pills">',
        '    <span class="ie-pill">Vision Language Model</span>',
        '    <span class="ie-pill">RAG · PubMed + PMC</span>',
        '    <span class="ie-pill">DICOM Native</span>',
        '    <span class="ie-pill">Echo · MRI · CT</span>',
        '  </div>',

        '  <button class="ie-btn" onclick="ieCloseSplash()">',
        '    Begin Analysis &nbsp;&#8594;',
        '  </button>',

        '  <div class="ie-disclaimer">',
        '    For research and evaluation use &nbsp;·&nbsp; Requires clinician verification',
        '  </div>',

        '</div>',
    ].join('');
    pd.body.appendChild(el);

    par.ieCloseSplash = function() {
        par.sessionStorage.setItem('ie-splash-seen', '1');
        var overlay = pd.getElementById('ie-splash');
        if (!overlay) return;
        overlay.style.animation = 'ieSplashOut .45s cubic-bezier(.22,.68,0,1.2) forwards';
        setTimeout(function() {
            if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
        }, 460);
    };
})();
</script>
""", height=0)


# ── Sidebar toggle JS ──────────────────────────────────────────
_components.html("""
<script>
(function() {
    var par = window.parent || window;
    var pd  = par.document;

    /* ── sidebar helpers ── */
    var isMobile = par.innerWidth <= 768;

    function _sbDispatch(btn) {
        btn.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true, view: par}));
    }

    function _sbIsOpen() {
        var sb = pd.querySelector('section[data-testid="stSidebar"]');
        return sb && sb.getBoundingClientRect().left >= -10;
    }

    function _sbSetOverlay(visible) {
        var overlay = pd.getElementById('sb-overlay');
        if (!overlay) {
            overlay = pd.createElement('div');
            overlay.id = 'sb-overlay';
            pd.body.appendChild(overlay);
            overlay.addEventListener('click', function() {
                overlay.style.display = 'none';
                var closeBtn = pd.querySelector('[data-testid="stSidebarNavCollapseButton"]');
                if (closeBtn) { _sbDispatch(closeBtn); return; }
                var sb = pd.querySelector('section[data-testid="stSidebar"]');
                if (sb) { sb.style.transition = 'transform 0.25s ease'; sb.style.transform = 'translateX(-100%)'; }
                /* show peek strip again after sidebar closes */
                setTimeout(function() {
                    var peek = pd.getElementById('sb-peek');
                    if (peek && par.innerWidth <= 768) peek.style.display = 'flex';
                }, 300);
            });
        }
        overlay.style.display = visible ? 'block' : 'none';
    }

    /* ── sidebar auto-open on desktop ── */
    if (!isMobile) {
        for (var key in localStorage) {
            if (key.toLowerCase().includes('sidebar')) localStorage.removeItem(key);
        }
        function tryOpen() {
            var btn = pd.querySelector('[data-testid="stSidebarCollapsedControl"]') ||
                      pd.querySelector('[data-testid="collapsedControl"]');
            if (btn) _sbDispatch(btn);
        }
        setTimeout(tryOpen, 300);
        setTimeout(tryOpen, 800);
    }

    if (!par._sbHamburgerWired) {
        par._sbHamburgerWired = true;
        pd.addEventListener('click', function(e) {
            var btn = (e.target.closest ? e.target.closest('#sb-hamburger') : null) ||
                      (e.target.id === 'sb-hamburger' ? e.target : null);
            if (!btn) return;
            var isOpen = _sbIsOpen();
            var toggle;
            if (isOpen) {
                toggle = pd.querySelector('[data-testid="stSidebarNavCollapseButton"]');
            } else {
                toggle = pd.querySelector('[data-testid="stSidebarCollapsedControl"]') ||
                         pd.querySelector('[data-testid="collapsedControl"]');
            }
            if (toggle) {
                _sbDispatch(toggle);
            } else {
                /* Fallback: directly slide the sidebar */
                var sb = pd.querySelector('section[data-testid="stSidebar"]');
                if (sb) {
                    sb.style.transition = 'transform 0.25s ease';
                    sb.style.transform = isOpen ? 'translateX(-100%)' : 'translateX(0)';
                }
            }
            /* Show/hide backdrop overlay on mobile */
            if (par.innerWidth <= 768) _sbSetOverlay(!isOpen);
        });
    }

    /* ── overlay auto-sync: hide overlay whenever sidebar closes via any means ── */
    if (!par._sbOverlaySyncId) {
        par._sbOverlaySyncId = setInterval(function() {
            if (par.innerWidth > 768) return;
            var overlay = pd.getElementById('sb-overlay');
            if (!overlay || overlay.style.display !== 'block') return;
            if (!_sbIsOpen()) overlay.style.display = 'none';
        }, 180);
    }

    /* ── instant overlay close when native sidebar << button is clicked ── */
    if (!par._sbNativeCloseWired) {
        par._sbNativeCloseWired = true;
        pd.addEventListener('click', function(e) {
            if (par.innerWidth > 768) return;
            var closeBtn = e.target.closest ? e.target.closest('[data-testid="stSidebarNavCollapseButton"]') : null;
            if (!closeBtn) return;
            var overlay = pd.getElementById('sb-overlay');
            if (overlay) overlay.style.display = 'none';
        });
    }

    /* ── mobile peek strip ── */
    (function() {
        if (par.innerWidth > 768) return;
        if (pd.getElementById('sb-peek')) return;

        var strip = pd.createElement('div');
        strip.id = 'sb-peek';
        strip.innerHTML =
            '<div class="peek-icon">&#128247;</div>' +   /* 📷 camera */
            '<div class="peek-icon">&#128194;</div>' +   /* 📂 folder */
            '<div class="peek-chevron">&#8250;</div>';   /* › expand */

        var css = pd.createElement('style');
        css.id = 'sb-peek-style';
        css.textContent = [
            '#sb-peek{',
            '  position:fixed;top:50%;left:0;transform:translateY(-50%);',
            '  width:44px;z-index:9998;',
            '  display:flex;flex-direction:column;align-items:center;gap:10px;',
            '  background:rgba(13,25,48,0.92);',
            '  border-radius:0 12px 12px 0;',
            '  padding:14px 0;',
            '  box-shadow:3px 0 16px rgba(0,0,0,0.45);',
            '  border-right:1px solid rgba(59,130,246,0.25);',
            '  border-top:1px solid rgba(59,130,246,0.18);',
            '  border-bottom:1px solid rgba(59,130,246,0.18);',
            '  cursor:pointer;transition:opacity .2s;',
            '}',
            '#sb-peek .peek-icon{font-size:17px;line-height:1;opacity:.75;}',
            '#sb-peek .peek-chevron{',
            '  font-size:22px;font-weight:300;color:#60A5FA;opacity:.9;',
            '  line-height:1;margin-top:2px;',
            '}'
        ].join('');
        pd.head.appendChild(css);
        pd.body.appendChild(strip);

        function peekSync() {
            if (par.innerWidth > 768) { strip.style.display = 'none'; return; }
            strip.style.display = _sbIsOpen() ? 'none' : 'flex';
        }
        peekSync();

        strip.addEventListener('click', function() {
            var btn = pd.querySelector('[data-testid="stSidebarCollapsedControl"]') ||
                      pd.querySelector('[data-testid="collapsedControl"]');
            if (btn) _sbDispatch(btn);
            else {
                var sb = pd.querySelector('section[data-testid="stSidebar"]');
                if (sb) { sb.style.transition = 'transform 0.25s ease'; sb.style.transform = 'translateX(0)'; }
            }
            _sbSetOverlay(true);
            setTimeout(peekSync, 350);
        });

        if (!par._sbPeekSyncId) {
            par._sbPeekSyncId = setInterval(peekSync, 200);
        }
        par.addEventListener('resize', peekSync);
    })();

    /* ── theme toggle ── */
    function syncIcons(dark) {
        var moon = pd.getElementById('icon-moon');
        var sun  = pd.getElementById('icon-sun');
        if (moon) moon.style.display = dark ? 'none'  : 'block';
        if (sun)  sun.style.display  = dark ? 'block' : 'none';
    }
    function applyTheme(dark) {
        if (dark) {
            pd.body.classList.add('dark-mode');
        } else {
            pd.body.classList.remove('dark-mode');
        }
        syncIcons(dark);
    }
    /* apply saved preference on every render */
    var savedTheme = localStorage.getItem('ie-theme');
    applyTheme(savedTheme === 'dark');

    if (!par._themeWired) {
        par._themeWired = true;
        pd.addEventListener('click', function(e) {
            var btn = e.target.closest ? e.target.closest('#theme-toggle') : null;
            if (!btn) return;
            var dark = !pd.body.classList.contains('dark-mode');
            localStorage.setItem('ie-theme', dark ? 'dark' : 'light');
            applyTheme(dark);
        });
    }
})();
</script>
""", height=0)


# ── Session state ──────────────────────────────────────────────
_defaults = {
    "messages":         [],
    "images":           [],
    "img_names":        [],
    "view_labels":      [],
    "cam_images":       [],
    "cam_names":        [],
    "cam_labels":       [],
    "modality":         "Echocardiogram",
    "history":          [],
    "last_findings":    None,
    "last_literature":  None,
    "last_modality":    "Echocardiogram",
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
        f"letter-spacing:.09em;color:#7EAAD4;margin:.9rem 0 .4rem;'>{text}</div>",
        unsafe_allow_html=True,
    )

with st.sidebar:

    # ── brand ──
    st.markdown(
        "<div id='sb-brand' style='padding-bottom:1.1rem;margin-bottom:.6rem;"
        "border-bottom:1px solid #2E4F7E;'>"
        "<div style='font-size:1.15rem;font-weight:700;color:#E0EEFF;"
        "letter-spacing:-.025em;'>Imaging<span style='color:#3B82F6'>.</span>Evidence</div>"
        "<div id='sb-brand-sub' style='font-size:.72rem;color:#7EAAD4;margin-top:.15rem;'>"
        "Multimodal cardiac imaging AI</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── new study ──
    if st.button("+ New study", use_container_width=True):
        for _k in ("messages","images","img_names","view_labels",
                   "cam_images","cam_names","cam_labels",
                   "last_findings","last_literature"):
            st.session_state[_k] = [] if isinstance(st.session_state[_k], list) else None
        st.session_state.pop("_cam_fp", None)
        st.rerun()

    # ── modality ──
    selected_modality = st.selectbox(
        "Modality",
        options=MODALITIES,
        index=MODALITIES.index(st.session_state.modality),
        format_func=lambda m: f"{MODALITY_ICONS.get(m,'')}  {m}",
        key="modality_select",
    )
    if selected_modality != st.session_state.modality:
        st.session_state.modality    = selected_modality
        st.session_state.images      = []
        st.session_state.img_names   = []
        st.session_state.view_labels = []
        st.rerun()

    modality = st.session_state.modality

    # ── upload ──
    uploaded_files = st.file_uploader(
        "upload",
        type=["jpg","jpeg","png","tiff","gif","mp4","avi","mov","webm","dcm","dicom"],
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
            f"<div style='font-size:.78rem;color:#3B82F6;font-weight:500;"
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

            if is_animated:
                title_suffix = f" · GIF · {n_total}f"
            elif is_video:
                title_suffix = f" · {Path(f.name).suffix.upper()[1:]}"
            elif is_dcm:
                title_suffix = " · DICOM"
            else:
                title_suffix = ""

            with st.expander(f"#{i+1} · {f.name[:24]}{title_suffix}", expanded=False):

                # ── DICOM ──────────────────────────────────────────────
                if is_dcm:
                    try:
                        dcm_frames = _dicom_frame_count(raw)
                        if dcm_frames > 1:
                            n_extract = st.slider(
                                "Frames to extract",
                                min_value=1, max_value=min(dcm_frames, 12),
                                value=min(4, dcm_frames), key=f"dcm_nf_{i}",
                            )
                            with st.spinner("Loading..."):
                                frames, total_f, meta = extract_dicom_frames(raw, n_extract)
                            thumb_cols = st.columns(len(frames))
                            for fi, (col, frame) in enumerate(zip(thumb_cols, frames)):
                                with col:
                                    st.image(frame, width="stretch", caption=f"Frame {fi+1}")
                            for fi, frame in enumerate(frames):
                                st.session_state.images.append(frame)
                                st.session_state.img_names.append(
                                    f"{f.name} [frame {fi+1}/{len(frames)}]"
                                )
                            n_added = len(frames)
                            st.caption(f"DICOM · {n_extract} of {total_f} frames")
                        else:
                            with st.spinner("Loading..."):
                                img, meta = _dicom_to_pil(raw, frame_idx=0)
                            st.image(img, width="stretch")
                            st.session_state.images.append(img)
                            st.session_state.img_names.append(f.name)
                            n_added = 1
                            st.caption("DICOM")
                    except Exception as e:
                        st.error(f"Could not read DICOM: {e}")
                        n_added = 0

                # ── ANIMATED GIF ───────────────────────────────────────
                elif is_animated:
                    st.image(raw, width="stretch")
                    n_extract = st.slider(
                        "Frames to extract",
                        min_value=1, max_value=min(n_total, 8),
                        value=min(4, n_total), key=f"gif_nf_{i}",
                    )
                    frames = extract_gif_frames(raw, n_extract)
                    for fi, frame in enumerate(frames):
                        st.session_state.images.append(frame)
                        st.session_state.img_names.append(
                            f"{f.name} [frame {fi+1}/{n_extract}]"
                        )
                    n_added = n_extract
                    st.caption(f"GIF · {n_extract} of {n_total} frames")

                # ── VIDEO ──────────────────────────────────────────────
                elif is_video:
                    st.video(raw)
                    n_extract = st.slider(
                        "Frames to extract",
                        min_value=1, max_value=8,
                        value=4, key=f"vid_nf_{i}",
                    )
                    with st.spinner("Loading..."):
                        try:
                            frames, n_total_vid = extract_video_frames(raw, f.name, n_extract)
                        except Exception as e:
                            st.error(f"Could not read video: {e}")
                            frames, n_total_vid = [], 0
                    if frames:
                        thumb_cols = st.columns(len(frames))
                        for fi, (col, frame) in enumerate(zip(thumb_cols, frames)):
                            with col:
                                st.image(frame, width="stretch", caption=f"Frame {fi+1}")
                        for fi, frame in enumerate(frames):
                            st.session_state.images.append(frame)
                            st.session_state.img_names.append(
                                f"{f.name} [frame {fi+1}/{len(frames)}]"
                            )
                    n_added = len(frames)
                    st.caption(f"Video · {n_added} frames extracted")

                # ── STATIC IMAGE ───────────────────────────────────────
                else:
                    try:
                        img = Image.open(io.BytesIO(raw)).convert("RGB")
                        st.session_state.images.append(img)
                        st.session_state.img_names.append(f.name)
                        st.image(img, width="stretch")
                        n_added = 1
                    except Exception as e:
                        st.error(f"Could not read image '{f.name}': {e}")
                        n_added = 0

                all_labels.extend([modality] * n_added)

        st.session_state.view_labels = all_labels

    # ── live capture ──
    with st.expander("📷 Capture with camera", expanded=False):
        _cap_tab, _rec_tab = st.tabs(["Photo", "Record video"])

        with _cap_tab:
            _cam_pic = st.camera_input(
                "Take frame", label_visibility="collapsed", key="cam_snapshot"
            )
            if _cam_pic is not None:
                _raw_cam = _cam_pic.getvalue()
                _fp = (len(_raw_cam), _raw_cam[:128])
                if st.session_state.get("_cam_fp") != _fp:
                    st.session_state["_cam_fp"] = _fp
                    _cam_img = Image.open(io.BytesIO(_raw_cam)).convert("RGB")
                    _cam_idx = len(st.session_state.cam_images) + 1
                    st.session_state.cam_images.append(_cam_img)
                    st.session_state.cam_names.append(f"camera_frame_{_cam_idx}.jpg")
                    st.session_state.cam_labels.append(modality)
            if st.session_state.cam_images:
                _n_cam = len(st.session_state.cam_images)
                st.markdown(
                    f"<div style='font-size:.78rem;color:#3B82F6;font-weight:500;"
                    f"margin:.3rem 0 .4rem;'>&#10003; {_n_cam} camera frame{'s' if _n_cam>1 else ''}</div>",
                    unsafe_allow_html=True,
                )
                if st.button("Clear camera frames", key="clear_cam", use_container_width=True):
                    st.session_state.cam_images = []
                    st.session_state.cam_names  = []
                    st.session_state.cam_labels = []
                    st.session_state.pop("_cam_fp", None)
                    st.rerun()

        with _rec_tab:
            st.caption("Record directly, then download and upload it above for analysis.")
            _components.html("""
<style>
#rec-wrap{font-family:-apple-system,BlinkMacSystemFont,'Inter',sans-serif;color:#E0EEFF;background:#0D1930;border-radius:8px;padding:10px;}
#rec-wrap video{width:100%;border-radius:6px;background:#142240;display:block;max-height:160px;object-fit:cover;}
.rec-btn{flex:1;padding:7px 6px;border-radius:6px;border:1px solid #2E4F7E;cursor:pointer;font-size:12px;font-weight:500;transition:background .12s;}
#startBtn{background:#1A3A7A;color:#60A5FA;border-color:#1E4D99;}
#startBtn:hover{background:#1E4D99;}
#stopBtn{background:#142240;color:#7EAAD4;}
#stopBtn:disabled{opacity:.4;cursor:default;}
#timer{text-align:center;color:#EF4444;font-size:11px;margin-top:4px;display:none;}
#dlBtn{display:none;margin-top:6px;padding:7px 10px;background:#142240;border:1px solid #2E4F7E;color:#E0EEFF;border-radius:6px;text-decoration:none;font-size:12px;text-align:center;width:100%;box-sizing:border-box;}
#hint{display:none;font-size:11px;color:#7EAAD4;margin-top:5px;text-align:center;line-height:1.5;}
</style>
<div id="rec-wrap">
  <video id="preview" autoplay muted playsinline></video>
  <div style="display:flex;gap:6px;margin-top:8px;">
    <button id="startBtn" class="rec-btn" onclick="startRec()">&#9679; Start Recording</button>
    <button id="stopBtn"  class="rec-btn" onclick="stopRec()" disabled>&#9632; Stop</button>
  </div>
  <div id="timer">&#9679; REC <span id="elapsed">0:00</span></div>
  <video id="playback" controls playsinline style="display:none;margin-top:8px;width:100%;border-radius:6px;max-height:140px;"></video>
  <a id="dlBtn" href="#" download="echo_recording.webm">&#11015; Download Recording</a>
  <div id="hint">Download complete → upload the file using the uploader above to analyze it.</div>
</div>
<script>
var _stream,_recorder,_chunks=[],_timer,_elapsed=0;
function getMime(){
  var t=['video/mp4;codecs=h264','video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'];
  return t.find(function(m){return MediaRecorder.isTypeSupported(m);})||'video/webm';
}
async function startRec(){
  try{
    _stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'},audio:false});
    document.getElementById('preview').srcObject=_stream;
    _chunks=[];_elapsed=0;
    _recorder=new MediaRecorder(_stream,{mimeType:getMime()});
    _recorder.ondataavailable=function(e){if(e.data.size>0)_chunks.push(e.data);};
    _recorder.onstop=finalize;
    _recorder.start(200);
    document.getElementById('startBtn').disabled=true;
    document.getElementById('stopBtn').disabled=false;
    document.getElementById('timer').style.display='block';
    _timer=setInterval(function(){
      _elapsed++;
      var m=Math.floor(_elapsed/60),s=_elapsed%60;
      document.getElementById('elapsed').textContent=m+':'+(s<10?'0':'')+s;
    },1000);
  }catch(e){alert('Camera error: '+e.message);}
}
function stopRec(){
  clearInterval(_timer);
  if(_recorder&&_recorder.state!=='inactive')_recorder.stop();
  if(_stream)_stream.getTracks().forEach(function(t){t.stop();});
  document.getElementById('startBtn').disabled=false;
  document.getElementById('stopBtn').disabled=true;
  document.getElementById('timer').style.display='none';
}
function finalize(){
  var mime=getMime(),ext=mime.includes('mp4')?'mp4':'webm';
  var blob=new Blob(_chunks,{type:mime}),url=URL.createObjectURL(blob);
  var pb=document.getElementById('playback');
  pb.src=url;pb.style.display='block';
  var dl=document.getElementById('dlBtn');
  dl.href=url;dl.download='echo_recording.'+ext;dl.style.display='block';
  document.getElementById('hint').style.display='block';
  document.getElementById('preview').style.display='none';
}
</script>
""", height=370)

    st.session_state["flow_selection"] = "Flow A"

    # ── multi-modality context indicator ──
    _prev_findings = st.session_state.get("last_findings")
    _prev_modality = st.session_state.get("last_modality", "")
    _is_adding     = bool(_prev_findings and _prev_modality and _prev_modality != modality)

    if _is_adding:
        _prev_icon = MODALITY_ICONS.get(_prev_modality, "🩻")
        _cur_icon  = MODALITY_ICONS.get(modality, "🩻")
        st.markdown(
            f"<div style='background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.22);"
            f"border-radius:8px;padding:.55rem .75rem;margin:.4rem 0 .5rem;'>"
            f"<div style='font-size:.72rem;font-weight:600;color:#60A5FA;"
            f"margin-bottom:.25rem;letter-spacing:.01em;'>Multi-modality study</div>"
            f"<div style='font-size:.75rem;color:#7EAAD4;line-height:1.55;'>"
            f"{_prev_icon} {_prev_modality} findings will be passed as context "
            f"to the {_cur_icon} {modality} analysis.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── study image counter ──
    _prev_user_msgs = [
        m for m in st.session_state.messages
        if m["role"] == "user" and m.get("images")
    ]
    _study_analyzed = sum(len(m.get("images", [])) for m in _prev_user_msgs)
    _cur_batch      = len(st.session_state.images) + len(st.session_state.cam_images)
    _grand_total    = _study_analyzed + _cur_batch

    if _grand_total > 0:
        _mod_counts: dict = {}
        for _m in _prev_user_msgs:
            _mm = _m.get("modality", "")
            _mod_counts[_mm] = _mod_counts.get(_mm, 0) + len(_m.get("images", []))

        _rows = "".join(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"font-size:.8rem;color:#B8D4F0;padding:.18rem 0;'>"
            f"<span>{MODALITY_ICONS.get(_mm,'🩻')} {_mm}</span>"
            f"<span style='font-weight:600;color:#E0EEFF;'>{_mc}</span></div>"
            for _mm, _mc in _mod_counts.items()
        )
        if _cur_batch > 0:
            _rows += (
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"font-size:.8rem;color:#60A5FA;padding:.18rem 0;'>"
                f"<span>{MODALITY_ICONS.get(modality,'🩻')} {modality}"
                f"<span style='font-size:.68rem;margin-left:.35rem;opacity:.75;'>ready</span></span>"
                f"<span style='font-weight:600;'>{_cur_batch}</span></div>"
            )

        st.markdown(
            f"<div style='background:#0D1930;border:1px solid #243F6A;"
            f"border-radius:8px;padding:.6rem .75rem;margin:.5rem 0;'>"
            f"<div style='font-size:.67rem;font-weight:600;text-transform:uppercase;"
            f"letter-spacing:.09em;color:#3D608A;margin-bottom:.38rem;'>Study images</div>"
            f"{_rows}"
            f"<div style='border-top:1px solid #1E3055;margin-top:.38rem;padding-top:.38rem;"
            f"display:flex;justify-content:space-between;font-size:.78rem;"
            f"color:#BFDBFE;font-weight:600;'>"
            f"<span>Total</span><span>{_grand_total} image{'s' if _grand_total != 1 else ''}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    _total_imgs = _cur_batch
    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

    _btn_label = f"Add {modality} to Study" if _is_adding else "Analyze"
    analyze_clicked = st.button(
        _btn_label,
        use_container_width=True,
        type="primary",
        disabled=_total_imgs == 0,
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
        f"background:#EBF2FF;border:1px solid #BFDBFE;color:#1447C0;"
        f"font-size:.72rem;font-weight:500;padding:.22rem .65rem;border-radius:999px;'>"
        f"<span style='width:6px;height:6px;background:#1A56DB;"
        f"border-radius:50%;display:inline-block;'></span>"
        f"{_label}</span>"
    )

st.markdown(
    f"<div id='ie-topbar' style='display:flex;align-items:center;justify-content:space-between;"
    f"padding:.8rem 0 1rem 0;border-bottom:1px solid var(--bd);margin-bottom:0;gap:.5rem;'>"
    f"<div style='display:flex;align-items:center;min-width:0;flex:1;overflow:hidden;'>"
    f"<button id='sb-hamburger' aria-label='Toggle menu'>"
    f"<svg width='20' height='20' viewBox='0 0 20 20' fill='none' xmlns='http://www.w3.org/2000/svg'>"
    f"<path d='M2.5 5h15M2.5 10h15M2.5 15h15' stroke='currentColor' stroke-width='1.75' stroke-linecap='round'/>"
    f"</svg></button>"
    f"<div style='font-size:1rem;font-weight:700;color:#1A56DB;letter-spacing:-.02em;"
    f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>"
    f"Imaging<span style='color:#3B82F6'>.</span>Evidence"
    f"<span style='font-size:.72rem;color:var(--mu);font-weight:400;margin-left:.6rem;'>"
    f"{MODALITY_ICONS.get(modality,'')} {modality}</span></div>"
    f"</div>"
    f"<div style='display:flex;align-items:center;gap:.4rem;flex-shrink:0;'>"
    f"<span id='topbar-status'>{_status}</span>"
    f"<button id='theme-toggle' title='Toggle dark / light mode'>"
    f"<svg id='icon-moon' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
    f"<path d='M21 12.79A9 9 0 1 1 11.21 3a7 7 0 0 0 9.79 9.79z'/></svg>"
    f"<svg id='icon-sun' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='display:none'>"
    f"<circle cx='12' cy='12' r='5'/><line x1='12' y1='1' x2='12' y2='3'/>"
    f"<line x1='12' y1='21' x2='12' y2='23'/><line x1='4.22' y1='4.22' x2='5.64' y2='5.64'/>"
    f"<line x1='18.36' y1='18.36' x2='19.78' y2='19.78'/><line x1='1' y1='12' x2='3' y2='12'/>"
    f"<line x1='21' y1='12' x2='23' y2='12'/><line x1='4.22' y1='19.78' x2='5.64' y2='18.36'/>"
    f"<line x1='18.36' y1='5.64' x2='19.78' y2='4.22'/></svg>"
    f"</button>"
    f"</div></div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════
#  CHAT / EMPTY STATE
# ══════════════════════════════════════════════════════════════
_AVATAR = {
    "user":      "🧑‍⚕️",
    "assistant": ":material/ecg_heart:",
}

if not st.session_state.messages:
    chips = EMPTY_STATE_CHIPS.get(modality, EMPTY_STATE_CHIPS["Echocardiogram"])
    st.markdown(
        "<div class='mobile-empty-state' style='text-align:center;padding:4.5rem 1rem 3rem;'>"
        f"<div style='font-size:2rem;opacity:.2;margin-bottom:1.2rem;'>{MODALITY_ICONS.get(modality,'🩻')}</div>"
        "<div style='font-size:1.45rem;font-weight:600;color:var(--ink);"
        "letter-spacing:-.025em;margin-bottom:.55rem;'>"
        f"Upload a {modality} study</div>"
        "<div style='font-size:.9rem;color:var(--mu);max-width:380px;"
        "margin:0 auto 2.5rem;line-height:1.65;'>"
        "Upload your imaging files, then tap <strong>Analyze</strong> to generate a clinical report."
        "</div>"
        "<div style='display:flex;flex-wrap:wrap;gap:.45rem;"
        "justify-content:center;max-width:520px;margin:0 auto;'>",
        unsafe_allow_html=True,
    )
    for chip in chips:
        st.markdown(
            f"<div style='background:var(--acl);border:1px solid var(--acm);color:var(--ac);"
            f"padding:.4rem .85rem;border-radius:999px;font-size:.82rem;'>{chip}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div></div>", unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        _av = _AVATAR.get(msg["role"], msg["role"])
        with st.chat_message(msg["role"], avatar=_av):
            if "images" in msg and msg["images"]:
                imgs = msg["images"]
                cols = st.columns(min(len(imgs), 4))
                for idx, (im, nm) in enumerate(zip(imgs, msg.get("img_names", []))):
                    with cols[idx % 4]:
                        st.image(im, caption=nm, width="stretch")

            st.markdown(msg["content"])

    _components.html("""
<script>
(function() {
    function doScroll() {
        var pd   = window.parent.document;
        var msgs = pd.querySelectorAll('[data-testid="stChatMessage"]');
        if (!msgs.length) return;
        /* Find the last user message — scroll to its top so the response
           appears just below, matching Claude-style reading flow */
        var target = null;
        for (var i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].querySelector('[data-testid="stChatMessageAvatarUser"]')) {
                target = msgs[i]; break;
            }
        }
        if (!target) target = msgs[msgs.length - 1];
        target.scrollIntoView({block: 'start', behavior: 'smooth'});
    }
    /* Two passes: first is fast (catches most cases),
       second overrides any late Streamlit auto-scroll */
    setTimeout(doScroll, 250);
    setTimeout(doScroll, 600);
})();
</script>
""", height=0)


# ══════════════════════════════════════════════════════════════
#  ANALYZE HANDLER
# ══════════════════════════════════════════════════════════════
if analyze_clicked and st.session_state.images:
    n        = len(st.session_state.images)
    modality = st.session_state.modality

    _ctx_findings = st.session_state.get("last_findings")
    _ctx_modality = st.session_state.get("last_modality", "")
    _is_multi     = bool(_ctx_findings and _ctx_modality and _ctx_modality != modality)

    if _is_multi:
        question = (
            f"This is a multi-modality cardiac study. The patient has already had a "
            f"{_ctx_modality} study with the following findings:\n\n"
            f"{_ctx_findings[:2000]}\n\n"
            f"Now analyze {'these' if n > 1 else 'this'} new {modality} "
            f"image{'s' if n > 1 else ''}. Cross-reference with the {_ctx_modality} findings above, "
            f"identify correlating or additional diagnostic information, and provide a "
            f"comprehensive {modality} report that builds on the multi-modality assessment."
        )
        _msg_content = (
            f"**{n} {modality} image{'s' if n > 1 else ''} added to study** "
            f"— cross-referencing with {_ctx_modality} findings."
        )
    else:
        question = (
            f"Please analyze {'these' if n > 1 else 'this'} {n} {modality} "
            f"image{'s' if n > 1 else ''} and provide a comprehensive clinical report."
        )
        _msg_content = f"**{n} {modality} image{'s' if n > 1 else ''} submitted for analysis.**"

    st.session_state.history.append({"label": modality, "n": n, "modality": modality})

    st.session_state.messages.append({
        "role":      "user",
        "content":   _msg_content,
        "images":    list(st.session_state.images),
        "img_names": list(st.session_state.img_names),
        "modality":  modality,
    })

    try:
        from agent import run_groq_vision, run_literature_search, run_synthesis

        findings   = None
        literature = None
        synthesis  = None

        with st.chat_message("assistant", avatar="⏳"):
            _status_label = f"Adding {modality} to study..." if _is_multi else "Analyzing..."
            with st.status(_status_label, expanded=True) as status:
                _review_msg = (
                    f"Reviewing {modality} images with {_ctx_modality} context..."
                    if _is_multi else f"Reviewing {modality} images..."
                )
                st.write(_review_msg)
                findings = run_groq_vision(st.session_state.images, question, modality)

                st.write("Searching medical literature...")
                literature = run_literature_search(findings, modality)

                st.write("Generating report...")
                synthesis = run_synthesis(findings, literature, modality)
                status.update(label="Done", state="complete", expanded=False)

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
                f"### {modality} Report — {n} image{'s' if n > 1 else ''}\n\n"
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
user_input = st.chat_input("Ask a follow-up question...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Show the user message immediately
    with st.chat_message("user", avatar=_AVATAR["user"]):
        st.markdown(user_input)

    last_findings   = st.session_state.get("last_findings")
    last_literature = st.session_state.get("last_literature")
    last_modality   = st.session_state.get("last_modality", "Echocardiogram")

    if last_findings:
        # Spinner avatar while answer is being generated
        with st.chat_message("assistant", avatar="⏳"):
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
                    result = f"**Error answering follow-up:** `{e}`\n```\n{traceback.format_exc()}\n```"
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": result,
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
    "<div style='text-align:center;color:var(--mu);font-size:.72rem;"
    "padding:1.5rem 0 .5rem;border-top:1px solid var(--bd);margin-top:2rem;line-height:1.9;'>"
    "Imaging Evidence &nbsp;·&nbsp; Research preview &nbsp;·&nbsp; "
    "Outputs require clinician verification before any clinical decision."
    "</div>",
    unsafe_allow_html=True,
)
