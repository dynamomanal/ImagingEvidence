import os
import io
import gc
import time
import ctypes
import warnings
import logging
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import transformers
from PIL import Image
import requests
import torch
from huggingface_hub import login

# ── Detect device once at import time ────────────────────────────────────────
import torch as _torch
DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    # SAFETENSORS_FAST_GPU intentionally NOT set — causes bfloat16 weights to
    # load to GPU before bitsandbytes can quantize them, which overflows VRAM
    # into CPU RAM and triggers the Windows OOM process kill.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# ── Suppress noisy warnings ───────────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
for _lg in ["transformers","accelerate","huggingface_hub","torch","bitsandbytes"]:
    logging.getLogger(_lg).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
logging.getLogger("huggingface_hub.utils._token").setLevel(logging.ERROR)
login(token=HF_TOKEN)


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT — baked into every MedGemma request
# ══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are a cardiac imaging AI. Analyze this echocardiogram and extract structured, measurable observations ONLY — do not interpret or diagnose.

Output exactly these sections:

**Image Quality**
[Good / Moderate / Poor — note any limiting factors]

**Left Ventricle**
- Size: [normal / mildly/moderately/severely dilated] — estimated LVIDd if visible
- Systolic function: estimate LVEF% if possible (e.g. "visually ~45%")
- Wall motion: [normal / describe any hypokinetic or akinetic segments with location]
- Wall thickness: [normal ~6-11 mm / thickened / thinned]

**Right Ventricle**
- Size: [normal / dilated]
- Function: [normal / reduced] — TAPSE estimate if visible

**Atria**
- Left atrium: [normal / mildly/moderately/severely dilated]
- Right atrium: [normal / dilated]

**Valves**
- Mitral: [normal / thickened / regurgitation / stenosis]
- Aortic: [normal / calcified / regurgitation / stenosis]
- Tricuspid / Pulmonic: [normal / abnormal]

**Pericardium**
[No effusion / small/moderate/large effusion]

**Other Observations**
[Any thrombus, mass, IVC dilation, or other notable finding]

Rules: If a structure is not visible or assessable, write "Not assessable".
Do not diagnose. If LVEF < 30% or cardiac tamponade is suspected, start with: URGENT FLAG: [finding]"""


# ══════════════════════════════════════════════════════════════
#  RAM PRE-FLIGHT CHECK
#  Converts a Windows OOM process-kill into a catchable Python
#  exception — the process survives and Streamlit shows an error.
# ══════════════════════════════════════════════════════════════

def _free_ram_gb() -> float:
    """Available physical RAM in GB (Windows ctypes, no extra dependency)."""
    try:
        class _MEMSTATEX(ctypes.Structure):
            _fields_ = [
                ("dwLength",                 ctypes.c_ulong),
                ("dwMemoryLoad",             ctypes.c_ulong),
                ("ullTotalPhys",             ctypes.c_ulonglong),
                ("ullAvailPhys",             ctypes.c_ulonglong),
                ("ullTotalPageFile",         ctypes.c_ulonglong),
                ("ullAvailPageFile",         ctypes.c_ulonglong),
                ("ullTotalVirtual",          ctypes.c_ulonglong),
                ("ullAvailVirtual",          ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        m = _MEMSTATEX(dwLength=ctypes.sizeof(_MEMSTATEX))
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 1024 ** 3
    except Exception:
        return 16.0   # assume OK if ctypes query fails


_RAM_REQUIRED_GB = 4.5   # after freeing background processes + with disk spill safety net


def _free_ram_for_loading():
    """
    Kill non-essential background processes and compact our own working set
    to give MedGemma loading the maximum headroom possible.
    SSD is used as overflow via offload_folder, so we only need a few GB free.
    """
    import subprocess, time

    # Background / tray processes that are safe to kill
    _TARGETS = [
        "SearchApp.exe", "SearchIndexer.exe",
        "OneDrive.exe",  "OneDriveSetup.exe",
        "Spotify.exe",   "SpotifyWebHelper.exe",
        "Discord.exe",
        "Teams.exe",     "ms-teams.exe",
        "slack.exe",
        "Dropbox.exe",   "GoogleDriveFS.exe",
        "AdobeUpdateService.exe", "AdobeARMservice.exe",
        "CompatTelRunner.exe",    "SgrmBroker.exe",
        "MicrosoftEdgeUpdate.exe","GoogleUpdate.exe",
        "nvtray.exe",             # NVIDIA system tray — restarts automatically
        "RAVBg64.exe",            # Realtek audio helper
        "igfxEM.exe",  "igfxHK.exe",   # Intel graphics helpers
        "TiWorker.exe",           # Windows Update worker
        "WmiPrvSE.exe",
    ]

    freed = []
    for proc in _TARGETS:
        r = subprocess.run(
            ["taskkill", "/F", "/IM", proc],
            capture_output=True,
        )
        if r.returncode == 0:
            freed.append(proc.replace(".exe", ""))

    if freed:
        print(f"[MODEL] Freed background processes: {', '.join(freed)}")
    else:
        print("[MODEL] No additional background processes found to free.")

    # Compact our own process working set — returns unused pages to the OS
    try:
        import ctypes
        ctypes.windll.psapi.EmptyWorkingSet(
            ctypes.windll.kernel32.GetCurrentProcess()
        )
    except Exception:
        pass

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    time.sleep(1.5)   # give OS a moment to reclaim the freed pages


def load_model():
    """
    Load MedGemma safely.

    Pre-flight: if free RAM < 6 GB raise MemoryError BEFORE touching
    from_pretrained.  This converts the silent Windows OOM process-kill
    into a Python exception that Streamlit can catch and display.

    GPU  — 4-bit NF4 via bitsandbytes (~2.5 GB VRAM after loading).
           offload_folder provides a disk-spill safety net so any
           unexpected overflow goes to disk instead of crashing.
    CPU  — float32 fallback (slow but functional).
    """
    MODEL_ID     = "google/medgemma-1.5-4b-it"
    OFFLOAD_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".offload_cache")

    # ── RAM check ────────────────────────────────────────────────────────────
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    free_gb = _free_ram_gb()
    print(f"[MODEL] Free RAM: {free_gb:.1f} GB  (need ≥{_RAM_REQUIRED_GB} GB)")

    if free_gb < _RAM_REQUIRED_GB:
        raise MemoryError(
            f"MedGemma needs at least {_RAM_REQUIRED_GB:.0f} GB free RAM, "
            f"but only {free_gb:.1f} GB is available.\n\n"
            "Close Chrome, VS Code, or other apps and try again, or use "
            "Flow A (Groq Vision) which needs no local GPU."
        )

    # ── Processor ────────────────────────────────────────────────────────────
    print("[MODEL] Loading processor…")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("[MODEL] Processor ready.")

    # ── Model ─────────────────────────────────────────────────────────────────
    if DEVICE == "cuda":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        # CPU cap at 2 GiB: anything that doesn't fit on GPU spills to SSD
        # (offload_folder), keeping system RAM free for Windows + Python.
        max_memory = {0: "5500MiB", "cpu": "2000MiB"}
        os.makedirs(OFFLOAD_DIR, exist_ok=True)

        print("[MODEL] Calling from_pretrained (GPU · 4-bit NF4 · SSD spill enabled)…")
        print("[MODEL] First load takes 3-8 min — model is cached after this…")
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            offload_state_dict=True,    # stream checkpoint one shard at a time
            offload_folder=OFFLOAD_DIR, # SSD spill — layers that exceed caps go here
        )
    else:
        print("[MODEL] Calling from_pretrained (CPU · float32 — slow)…")
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            device_map="cpu",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )

    model.eval()
    print(f"[MODEL] Model ready · device: {DEVICE}")
    return {"model": model, "processor": processor, "device": DEVICE}


def analyze_image(model_bundle: dict, images: list,
                  user_question: str = "Analyze this echocardiogram.") -> str:
    """
    Run MedGemma on one or more images using the two-step processor pattern:
      1. apply_chat_template(..., tokenize=False) — text with <image> placeholders
      2. processor(text, images) — tokenise + encode pixels together
    """
    model     = model_bundle["model"]
    processor = model_bundle["processor"]

    MAX_SIDE = 336
    resized  = []
    for img in images:
        w, h  = img.size
        scale = min(MAX_SIDE / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        resized.append(img)

    content  = [{"type": "image"} for _ in resized]
    content.append({"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{user_question}"})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=text, images=resized, return_tensors="pt", padding=True
    ).to(DEVICE)

    if "pixel_values" in inputs and inputs["pixel_values"].dtype != torch.bfloat16:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=500, do_sample=False)

    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True)


if __name__ == "__main__":
    bundle   = load_model()
    resp     = requests.get(
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
        headers={"User-Agent": "test"},
    )
    image    = Image.open(io.BytesIO(resp.content)).convert("RGB")
    print(analyze_image(bundle, [image]))
