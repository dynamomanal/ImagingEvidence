"""
3-Stage RAG Pipeline
====================
Stage 1  ->  Groq Vision (Llama 4 Scout) or MedGemma
             -- image -> structured modality-specific descriptions
Stage 2  ->  PubMed + Europe PMC + Semantic Scholar (24 papers)
             -- fed into TF-IDF vector store, top-12 retrieved per query
Stage 3  ->  Groq / LLaMA-3
             -- OpenEvidence-style report: clinical summary, key findings,
               differential, next steps, full numbered references

Supported modalities:
  Echo | Cardiac MRI | Cardiac CT | Nuclear (SPECT/PET) | Chest X-Ray | Coronary Angiography
"""

import os
import base64
from io import BytesIO
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# On Streamlit Cloud, secrets live in st.secrets — inject into os.environ
# so all os.getenv() calls work the same way locally and in the cloud.
try:
    import streamlit as st
    for _k in ("GROQ_API_KEY", "HF_TOKEN", "HF_PWD"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass


# ══════════════════════════════════════════════════════════════
#  STAGE 1 — Modality-specific vision system prompts
# ══════════════════════════════════════════════════════════════

VISION_SYSTEMS = {

"Echo": """You are a cardiac imaging AI. Analyze this echocardiogram and extract structured, measurable observations ONLY — do not interpret or diagnose.

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
- Mitral: [normal / thickened / regurgitation / stenosis — describe]
- Aortic: [normal / calcified / regurgitation / stenosis — describe]
- Tricuspid / Pulmonic: [normal / abnormal — describe]

**Pericardium**
[No effusion / small/moderate/large effusion]

**Other Observations**
[Any thrombus, mass, IVC dilation, or other notable finding]

Rules:
- If a structure is not visible or assessable, write "Not assessable".
- Do not diagnose or recommend treatment.
- If LVEF appears < 30% or cardiac tamponade is suspected, start the response with: URGENT FLAG: [finding]""",


"Cardiac MRI": """You are a cardiac MRI (CMR) reporting AI. Extract structured, measurable observations from this CMR image — do not diagnose.

Output exactly these sections:

**Image Type & Quality**
[Sequence type if identifiable: cine / LGE / T1 mapping / T2 mapping / STIR / perfusion / phase-contrast — Good / Moderate / Poor quality]

**Left Ventricle** (if cine or LGE)
- EDV / ESV / EF: [estimate if cine; state "Not assessable" for non-cine sequences]
- Wall motion: [normal / describe regional hypokinesis or akinesis by segment name]
- Wall thickness: [normal / thinned / hypertrophied — segment if focal]
- LGE pattern (if LGE sequence): [none / subendocardial / midwall / epicardial / transmural — location and estimated % myocardial involvement]

**Right Ventricle** (if visible)
- Size and function: [normal / dilated / reduced RVEF estimate]

**Atria**
- Left atrium: [size — normal / dilated]
- Right atrium: [size]

**Myocardial Tissue Characterization**
- T1 / T2 signal (if mapping sequence): [elevated / normal / reduced — location]
- Edema (T2 / STIR): [present / absent — location]
- Fibrosis / LGE: [pattern and extent as described above]

**Pericardium**
[Normal / thickened / effusion / pericardial LGE]

**Valves & Great Vessels**
[Aortic root size if visible; any regurgitant jet if phase-contrast; aortic dimensions]

**Other Observations**
[Thrombus, mass, congenital anomaly, incidental finding]

Rules:
- If a structure is not visible, write "Not assessable".
- Do not diagnose or recommend treatment.
- If transmural LGE >50% in multiple segments is visible, start with: URGENT FLAG: [finding]""",


"Cardiac CT": """You are a cardiac CT (CCTA / calcium score / structural) reporting AI. Extract structured, measurable observations — do not diagnose.

Output exactly these sections:

**Image Type & Quality**
[CCTA / Calcium Score / TAVR planning / Pulmonary veins / Aorta — heart rate if relevant — motion artifact: none / mild / severe — Good / Moderate / Poor]

**Coronary Arteries**
- Dominance: [right / left / co-dominant]
- Left Main (LM): [normal / stenosis % / plaque: calcified / non-calcified / mixed]
- LAD: [proximal / mid / distal — stenosis %, plaque type]
- LCX: [same format]
- RCA: [same format]
- Agatston Calcium Score: [value — or "Not performed"]

**Left Ventricle**
- Size and wall thickness: [normal / dilated / hypertrophied]
- Myocardial enhancement / hypodensity: [none / describe]

**Aorta & Great Vessels**
- Aortic root diameter: [mm if visible]
- Ascending / arch / descending: [normal / dilated / dissection flap / calcification]

**Pericardium**
[Normal / effusion — size / calcification / thickening]

**Pulmonary Arteries**
[Normal / filling defect suggesting PE / dilation]

**Incidental Findings**
[Lung nodules, mediastinal lymph nodes, pleural effusion, hepatic, other]

Rules:
- If a structure is not visible or not imaged, write "Not assessable".
- Do not diagnose or recommend treatment.
- If critical stenosis (>70%) of LM or proximal LAD is suspected, or aortic dissection is visible, start with: URGENT FLAG: [finding]""",


"Nuclear (SPECT/PET)": """You are a nuclear cardiology reporting AI (SPECT MPI / PET). Extract structured observations — do not diagnose.

Output exactly these sections:

**Study Type & Quality**
[Exercise stress / pharmacologic stress / rest only — SPECT / PET — Tracer: Tc-99m sestamibi / Tl-201 / Rb-82 / FDG — Image quality: Good / Moderate / Poor — Patient motion: none / present]

**Stress Perfusion**
- Defect present: [yes / no]
- If yes — Location (territory): [LAD / LCX / RCA / multi-vessel]
- Size: [small <10% / medium 10-20% / large >20% LV]
- Severity: [mild / moderate / severe reduction in tracer uptake]
- Distribution: [subendocardial / transmural]

**Rest Perfusion**
- Defect present: [yes / no — same territory or different]
- Reversibility vs stress: [fully reversible = ischemia / partially reversible = mixed / fixed = scar]

**Gated SPECT / PET Wall Motion** (if available)
- LVEF (gated): [% — normal >50%]
- Wall motion: [normal / regional hypokinesis or akinesis — territory]
- Wall thickening: [normal / reduced in defect territory]

**Viability (PET FDG if applicable)**
- Perfusion-metabolism mismatch: [present / absent — location — implies hibernating myocardium]
- Match (perfusion-metabolism): [scar]

**Extracardiac Findings**
[Elevated lung uptake, hepatic uptake, incidental finding]

Rules:
- If a structure is not assessable, write "Not assessable".
- Do not diagnose or recommend treatment.
- If a large anterior / LAD-territory fixed defect with visually reduced EF is seen, start with: URGENT FLAG: [finding]""",


"Chest X-Ray": """You are a chest radiograph reporting AI focused on cardiac and cardiopulmonary findings. Extract structured observations — do not diagnose.

Output exactly these sections:

**Image Quality & Projection**
[PA / AP (portable) / Lateral — Inspiration: adequate / inadequate — Rotation: none / mild — Quality: Good / Moderate / Poor]

**Cardiac Silhouette**
- Cardiothoracic ratio (CTR): [estimate — normal <0.5 on PA / not reliable on AP]
- Cardiac contour: [normal / globally enlarged / specific chamber prominence if apparent]
- Aortic knuckle: [normal / prominent / calcified]

**Pulmonary Vasculature**
- Vascular markings: [normal / increased / decreased / cephalization / interstitial edema pattern / alveolar edema / Kerley B lines]
- Main pulmonary artery: [normal / enlarged — suggests pulmonary hypertension]

**Pleural Space**
- Effusion: [none / small / moderate / large — bilateral / left / right — blunting of costophrenic angle]
- Pneumothorax: [absent / present — size]

**Lung Fields**
- Consolidation / infiltrate: [none / location and pattern]
- Hyperinflation / air trapping: [none / present]
- Pulmonary edema: [none / interstitial / alveolar]

**Mediastinum**
- Width: [normal / widened]
- Hilar enlargement or adenopathy: [none / present]

**Devices & Bones**
[Pacemaker / ICD leads — prosthetic valves — rib notching — vertebral changes — incidental]

Rules:
- If a structure is not visible, write "Not assessable".
- Do not diagnose or recommend treatment.
- If acute pulmonary edema, tension pneumothorax, or widened mediastinum suggesting dissection is visible, start with: URGENT FLAG: [finding]""",


"Coronary Angiography": """You are a coronary angiography reporting AI. Extract structured, measurable observations per vessel and projection — do not diagnose.

Output exactly these sections:

**Image Quality & Projection**
[Projection angle — Vessel in view — Contrast fill: adequate / inadequate — Good / Moderate / Poor]

**Vessel Findings**
For each vessel visible:
- Vessel: [LM / LAD / Diagonal / LCX / OM / Ramus / RCA / PDA / PL]
- Stenosis: [none / % stenosis — location: proximal / mid / distal]
- Lesion morphology: [concentric / eccentric / calcified / soft / thrombus / dissection / bifurcation involvement]
- TIMI flow: [0 / 1 / 2 / 3]
- Collaterals: [none / Rentrop grade — recipient territory]

**Intervention (if PCI visible)**
[Stent placement — vessel / location / size if visible — result: TIMI 3 / residual stenosis]

**Other Observations**
[SCAD pattern, anomalous origin, ectasia, slow flow, no-reflow, coronary spasm]

Rules:
- If a vessel is not visualized, write "Not assessed".
- Do not diagnose or recommend treatment.
- If TIMI 0 or 1 flow is present in a major epicardial vessel, start with: URGENT FLAG: TIMI [grade] flow — [vessel]""",

}

# Backward-compat alias used by MedGemma flow
VISION_SYSTEM = VISION_SYSTEMS["Echo"]


# ── Modality -> PubMed search suffix ─────────────────────────────────────────
_MODALITY_SUFFIX = {
    "Echo":                   "echocardiography",
    "Cardiac MRI":            "cardiac MRI CMR cardiovascular magnetic resonance",
    "Cardiac CT":             "cardiac CT CCTA coronary computed tomography",
    "Nuclear (SPECT/PET)":   "nuclear cardiology SPECT myocardial perfusion imaging",
    "Chest X-Ray":            "chest radiograph chest X-ray cardiac",
    "Coronary Angiography":   "coronary angiography interventional cardiology",
}


def _pil_to_b64(img) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _tile_images(images: list, size: int = 336, max_cols: int = 2):
    if len(images) == 1:
        return images[0]
    cols = min(len(images), max_cols)
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * size, rows * size), (20, 20, 20))
    for i, img in enumerate(images):
        r, c  = divmod(i, cols)
        thumb = img.resize((size, size), Image.LANCZOS)
        grid.paste(thumb, (c * size, r * size))
    return grid


def run_groq_vision(images: list, question: str, modality: str = "Echo") -> str:
    """Stage 1: Groq vision model with modality-specific system prompt."""
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return "_Groq API key not configured. Add GROQ_API_KEY to .env._"

    system_prompt = VISION_SYSTEMS.get(modality, VISION_SYSTEMS["Echo"])

    tiled = _tile_images(images)
    b64   = _pil_to_b64(tiled)

    client  = Groq(api_key=api_key)
    content = [
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text",
         "text": f"{system_prompt}\n\n{question}"},
    ]

    for model_id in [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.2-11b-vision-preview",
    ]:
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": content}],
                max_tokens=700,
                temperature=0.1,
            )
            return resp.choices[0].message.content
        except Exception:
            continue

    return "_Vision analysis failed — check Groq API key or model availability._"


# ── Local MedGemma (Flow B) ───────────────────────────────────────────────────

_model_bundle = None

def _get_model():
    global _model_bundle
    if _model_bundle is None:
        from model import load_model
        _model_bundle = load_model()
    return _model_bundle

def run_medgemma(model_bundle: dict, images: list, question: str) -> str:
    from model import analyze_image
    return analyze_image(model_bundle, images, question)


# ══════════════════════════════════════════════════════════════
#  STAGE 2 — Literature retrieval + vector store
# ══════════════════════════════════════════════════════════════

_CONDITION_MAP = [
    # Echo / general cardiac
    (r"dilated\s+cardiomyopathy|DCM\b",                       '"dilated cardiomyopathy"'),
    (r"hypertrophic\s+cardiomyopathy|HCM\b|LVOTO|outflow\s+tract\s+obstruction",
                                                               '"hypertrophic cardiomyopathy"'),
    (r"ischemic\s+cardiomyopathy|coronary\s+artery\s+disease|regional\s+wall\s+motion",
                                                               '"ischemic cardiomyopathy"'),
    (r"HFrEF|heart\s+failure\s+with\s+reduced|LVEF\s*[~<]\s*[1-3][0-9]|EF.*\b[1-2]\d\b",
                                                               '"heart failure with reduced ejection fraction"'),
    (r"HFpEF|diastolic\s+dysfunction|preserved\s+ejection\s+fraction",
                                                               '"heart failure with preserved ejection fraction"'),
    (r"aortic\s+stenosis|calcified\s+aortic|AVA\b|aortic\s+valve\s+area",
                                                               '"aortic stenosis"'),
    (r"aortic\s+regurgitation|aortic\s+insufficiency",        '"aortic regurgitation"'),
    (r"mitral\s+stenosis|rheumatic\s+mitral|MVA\b",           '"mitral stenosis"'),
    (r"mitral\s+regurgitation|mitral\s+insufficiency|\bMR\b", '"mitral regurgitation"'),
    (r"tricuspid\s+regurgitation|\bTR\b",                     '"tricuspid regurgitation"'),
    (r"pericardial\s+effusion",                               '"pericardial effusion"'),
    (r"cardiac\s+tamponade|tamponade",                        '"cardiac tamponade"'),
    (r"pulmonary\s+hypertension|elevated\s+RVSP|RV\s+pressure",
                                                               '"pulmonary hypertension"'),
    (r"RV\s+dysfunction|right\s+ventricular\s+fail|cor\s+pulmonale",
                                                               '"right ventricular dysfunction"'),
    (r"amyloid",                                              '"cardiac amyloidosis"'),
    (r"endocarditis|vegetation",                              '"infective endocarditis"'),
    (r"Takotsubo|apical\s+ballooning|stress\s+cardiomyopathy",
                                                               '"Takotsubo cardiomyopathy"'),
    (r"LVNC|non.compaction",                                  '"left ventricular non-compaction"'),
    (r"intracardiac\s+thrombus|LV\s+thrombus",                '"left ventricular thrombus"'),
    (r"biventricular\s+fail",                                 '"biventricular heart failure"'),

    # Cardiac MRI-specific
    (r"late\s+gadolinium|LGE|gadolinium\s+enhancement",       '"late gadolinium enhancement" CMR'),
    (r"T1\s+mapping|native\s+T1|ECV",                        '"T1 mapping" cardiac MRI'),
    (r"T2\s+mapping|myocardial\s+edema|STIR",                '"T2 mapping" myocarditis CMR'),
    (r"myocarditis",                                          '"myocarditis" cardiac MRI'),
    (r"arrhythmogenic|ARVC|ARVD",                             '"arrhythmogenic right ventricular cardiomyopathy"'),
    (r"iron\s+overload|hemosiderosis",                        '"myocardial iron overload" CMR'),

    # Cardiac CT-specific
    (r"calcium\s+score|Agatston|coronary\s+calcif",           '"coronary artery calcium score"'),
    (r"CCTA|coronary\s+CT\s+angiography|CTA\s+coronary",     '"coronary CT angiography"'),
    (r"left\s+main\s+stenosis|LM\s+stenosis",                '"left main coronary artery stenosis"'),
    (r"LAD\s+stenosis|proximal\s+LAD",                        '"LAD stenosis" coronary'),
    (r"aortic\s+dissection|dissection\s+flap",               '"aortic dissection"'),
    (r"pulmonary\s+embolism|PE\b|filling\s+defect.*pulm",    '"pulmonary embolism" CT'),
    (r"TAVR|transcatheter\s+aortic",                         '"transcatheter aortic valve replacement"'),

    # Nuclear-specific
    (r"reversible\s+defect|ischemi.*perfusion|perfusion.*defect",
                                                               '"myocardial perfusion imaging" SPECT ischemia'),
    (r"fixed\s+defect|myocardial\s+scar|infarct.*perfusion",
                                                               '"myocardial scar" nuclear perfusion'),
    (r"hibernat|viabilit.*PET|FDG.*viabilit",                '"myocardial hibernation" PET viability'),
    (r"mismatch.*perfusion|perfusion.*mismatch",             '"perfusion metabolism mismatch" PET'),

    # Chest X-Ray-specific
    (r"pulmonary\s+edema|alveolar\s+edema|Kerley",           '"pulmonary edema" chest radiograph'),
    (r"pleural\s+effusion.*bilateral|bilateral.*pleural",    '"bilateral pleural effusion" heart failure'),
    (r"cardiomegaly|enlarged\s+cardiac\s+silhouette",        '"cardiomegaly" chest X-ray'),

    # Angiography-specific
    (r"TIMI\s+[012]|no.reflow|slow\s+flow",                  '"TIMI flow" coronary angiography'),
    (r"SCAD|spontaneous\s+coronary\s+artery\s+dissection",   '"spontaneous coronary artery dissection"'),
    (r"coronary\s+spasm|vasospasm",                          '"coronary vasospasm"'),
    (r"stent\s+thrombosis|in.stent\s+restenosis",            '"stent thrombosis" "in-stent restenosis"'),
]


def _build_search_query(findings: str, modality: str = "Echo") -> str:
    """
    Map detected cardiac conditions to precise PubMed search terms.
    Falls back to raw clinical keywords with the appropriate modality suffix.
    """
    import re
    text = findings[:800]
    suffix = _MODALITY_SUFFIX.get(modality, "cardiology")

    matched = []
    for pattern, term in _CONDITION_MAP:
        if re.search(pattern, text, flags=re.IGNORECASE):
            matched.append(term)

    if matched:
        query = " OR ".join(matched[:3])
        return f"({query}) AND {suffix}"

    # Fallback — extract raw clinical terms
    terms = re.findall(
        r"(?:LVEF|EF|LVIDd|TAPSE|cardiomyopathy|hypokinesis|akinesis|"
        r"regurgitation|stenosis|effusion|tamponade|heart failure|"
        r"ejection fraction|wall motion|biventricular|RV dysfunction|"
        r"LGE|T1 mapping|T2 mapping|calcium score|perfusion defect|"
        r"ischemic|non-ischemic|DCM|HFrEF|HFpEF|TIMI|dissection)[^,;\n]{0,40}",
        text,
        flags=re.IGNORECASE,
    )
    if terms:
        return " ".join(t.strip() for t in terms[:4]) + f" {suffix}"

    return f"{text[:200]} {suffix}"


def run_literature_search(findings: str, modality: str = "Echo") -> dict:
    """
    Query PubMed (15), Europe PMC (8), and Semantic Scholar (6).
    All papers are ingested into the vector store.
    Returns raw dict keyed by source; synthesis re-queries store for top-12.
    """
    from tools import fetch_pubmed_articles, fetch_pmc_articles, fetch_semantic_scholar_articles
    from vector_store import get_store

    query = _build_search_query(findings, modality)

    pubmed_articles  = fetch_pubmed_articles(query, max_results=15)
    pmc_articles     = fetch_pmc_articles(query, max_results=8)
    scholar_articles = fetch_semantic_scholar_articles(query, max_results=6)

    store = get_store()
    added_pub  = store.add_papers(pubmed_articles)
    added_pmc  = store.add_papers(pmc_articles)
    added_sch  = store.add_papers(scholar_articles)
    mode = "semantic" if store.using_semantic else "TF-IDF"
    print(f"[STORE] Added {added_pub} PubMed + {added_pmc} PMC + {added_sch} Scholar "
          f"-> {store.count} total papers ({mode} retrieval)")

    return {
        "pubmed":  pubmed_articles,
        "pmc":     pmc_articles,
        "scholar": scholar_articles,
    }


# ══════════════════════════════════════════════════════════════
#  STAGE 3 — OpenEvidence-style synthesis
# ══════════════════════════════════════════════════════════════

SYNTHESIS_SYSTEM = """You are a clinical cardiology AI that produces an OpenEvidence-style cardiac imaging report.

You receive:
  1. Structured imaging descriptions (Stage 1 — modality-specific measurements and observations)
  2. Numbered research abstracts from PubMed / PMC / Semantic Scholar
  3. The imaging modality that was used

Adapt your report structure to the modality:
- Echo: focus on LV/RV function, valves, haemodynamics
- Cardiac MRI: focus on volumes, LGE pattern, tissue characterization
- Cardiac CT: focus on coronary stenosis, calcium score, structural findings
- Nuclear (SPECT/PET): focus on perfusion territory, reversibility, viability
- Chest X-Ray: focus on cardiac silhouette, pulmonary vasculature, pleural findings
- Coronary Angiography: focus on stenosis severity, TIMI flow, territory at risk

Follow this EXACT format — do not deviate:

---

[Opening paragraph — NO heading]
2-3 sentences. Name the overall syndrome in **bold**. Name the 1-2 most likely diagnoses in **bold**. End with what workup distinguishes them.

---

**Summary of Key Imaging Findings**

6-8 bullet points, each:
- **[Finding name]** — clinical meaning, quantitative context vs normal range, and cite with [N]

---

**Differential Diagnosis**

Most likely:
- **[Top diagnosis]** — reasoning with sensitivity/specificity or % prevalence from the literature. [N][N]
- **[Second diagnosis]** — brief reasoning. [N]

Important not to miss:
- **[Diagnosis 3]** — how it mimics this picture; distinguishing imaging feature. [N]
- **[Diagnosis 4 if applicable]** — brief note. [N]

---

**Recommended Next Steps**

One sentence of setup context.

**Targeted history:** 3-4 specific questions that meaningfully shift the differential.

**Initial workup:**
- **[Specific test]** — why indicated; guideline class/recommendation if available. [N]
- **[Specific test]** — [N]
(list 4-6 tests)

End with 2-3 sentences on treatment implications relevant to the modality findings. [N]

---

Is there additional clinical context that would help narrow the differential?

### References

Number every cited paper in order of first appearance:
1. [Authors] ([YEAR]). [Title]. *[Journal]*. doi:[DOI]. [URL]
2. ...

---

STRICT RULES:
- Every number, range, threshold, sensitivity/specificity, and guideline statement MUST carry an inline [N] citation.
- Cite ONLY papers from the numbered list provided — never fabricate statistics.
- Bold key diagnoses, findings, and test names.
- Include every cited paper in the References section.
- If fewer than 4 papers have usable abstracts, note limited literature.
- Target 700-950 words total."""


def run_synthesis(findings: str, literature: dict, modality: str = "Echo") -> str:
    """
    Re-query the vector store for top-12 relevant papers,
    build a numbered prompt, and call Groq LLaMA-3 for synthesis.
    """
    from groq import Groq
    from vector_store import get_store

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return (
            "_Groq API key not configured. Add `GROQ_API_KEY` to your .env file._\n\n"
            f"**Image Findings:**\n{findings}"
        )

    store   = get_store()
    papers  = store.search(findings, top_k=12)

    if not papers:
        for src in ("pubmed", "pmc", "scholar"):
            for a in literature.get(src, []):
                if a.get("abstract"):
                    papers.append(a)
        papers = papers[:12]

    papers = [p for p in papers if p.get("abstract")]

    if papers:
        lit_block = ""
        for i, a in enumerate(papers, 1):
            authors = a.get("authors", "")
            doi_str = f"doi:{a['doi']}  " if a.get("doi") else ""
            lit_block += (
                f"[{i}] {authors}\n"
                f"    Title: {a['title']}\n"
                f"    Journal: {a['journal']} ({a['year']})\n"
                f"    Abstract: {a['abstract']}\n"
                f"    {doi_str}Link: {a['link']}\n\n"
            )
    else:
        lit_block = "No abstracts retrieved. Produce report with general cardiology knowledge, noting limited literature."

    user_message = (
        f"## Imaging Modality\n{modality}\n\n"
        "## Image Descriptions (Stage 1 — Vision Model)\n"
        "Use the observations below as the factual basis for the report. "
        "Every measurement or finding you discuss must appear in these descriptions.\n\n"
        f"{findings}\n\n"
        "---\n\n"
        "## Retrieved Literature (cite as [N] inline)\n"
        "Use ONLY data from these abstracts. Do not fabricate statistics.\n\n"
        f"{lit_block}"
    )

    client = Groq(api_key=api_key)
    for _model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
        try:
            response = client.chat.completions.create(
                model=_model,
                messages=[
                    {"role": "system", "content": SYNTHESIS_SYSTEM},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=1400,
                temperature=0.15,
            )
            return response.choices[0].message.content
        except Exception as _e:
            last_err = _e
            continue
    return (
        f"_Report synthesis unavailable ({last_err}). "
        "Raw image findings below:_\n\n"
        f"{findings}"
    )


# ══════════════════════════════════════════════════════════════
#  FOLLOW-UP Q&A  — uses stored findings, no re-vision
# ══════════════════════════════════════════════════════════════

FOLLOWUP_SYSTEM = """You are a clinical cardiology AI answering a specific follow-up question about a cardiac imaging case.

You are given:
  1. Structured imaging findings from a prior vision analysis (modality noted)
  2. Numbered research papers retrieved from PubMed / PMC / Semantic Scholar
  3. Prior conversation history for context

Answer rules:
- Answer the specific question directly — be concise (150-350 words)
- Cite every statistic, risk figure, or guideline statement with [N] from the numbered papers
- Link your answer to the specific imaging findings when relevant
- Use clear clinical language accessible to a cardiology fellow
- Do NOT repeat the full imaging report — answer only what is asked
- End with a **Sources cited:** section listing only papers you actually referenced
- If the provided literature does not directly address the question, say so honestly and answer from general cardiology knowledge, noting the limitation

Never fabricate statistics. Cite only from the provided paper list."""


def answer_followup(
    question: str,
    findings: str,
    literature: dict,
    history: list,
    modality: str = "Echo",
) -> str:
    """
    Answer a follow-up question using stored imaging findings and the cached paper store.
    Does NOT re-run the vision model.
    """
    from groq import Groq
    from vector_store import get_store

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return "_Groq API key not configured. Add GROQ_API_KEY to .env._"

    store = get_store()
    search_query = f"{question} {findings[:200]}"
    papers = store.search(search_query, top_k=8)

    if not papers:
        flat = []
        for src in ("pubmed", "pmc", "scholar"):
            flat.extend(a for a in (literature or {}).get(src, []) if a.get("abstract"))
        papers = flat[:8]

    papers = [p for p in papers if p.get("abstract")]

    lit_block = ""
    for i, a in enumerate(papers, 1):
        doi_str   = f"doi:{a['doi']}  " if a.get("doi") else ""
        score_str = f"  [relevance: {a['score']:.3f}]" if a.get("score") else ""
        lit_block += (
            f"[{i}] {a.get('authors', 'Unknown authors')}\n"
            f"    Title: {a['title']}\n"
            f"    Journal: {a.get('journal','N/A')} ({a.get('year','N/A')})\n"
            f"    Abstract: {a.get('abstract','')}\n"
            f"    {doi_str}Link: {a.get('link','N/A')}{score_str}\n\n"
        )

    history_text = ""
    recent = [m for m in (history or []) if m.get("role") in ("user", "assistant")][-6:-1]
    for msg in recent:
        role    = "Clinician" if msg["role"] == "user" else "AI"
        snippet = msg.get("content", "")[:300].replace("\n", " ")
        history_text += f"**{role}:** {snippet}...\n\n"

    user_message = (
        f"## Imaging Modality\n{modality}\n\n"
        "## Imaging Findings (from initial analysis)\n"
        f"{findings[:900]}\n\n"
        "---\n\n"
        f"## Prior Conversation\n"
        f"{history_text or '_No prior conversation._'}\n\n"
        "---\n\n"
        "## Retrieved Literature (cite as [N])\n"
        f"{lit_block or '_No papers retrieved for this query._'}\n\n"
        "---\n\n"
        f"## Question\n{question}"
    )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": FOLLOWUP_SYSTEM},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=700,
        temperature=0.15,
    )
    return response.choices[0].message.content


# ══════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ══════════════════════════════════════════════════════════════

def run_pipeline(images: list, question: str, modality: str = "Echo") -> dict:
    findings = run_groq_vision(images, question, modality)
    lit      = run_literature_search(findings, modality)
    report   = run_synthesis(findings, lit, modality)
    return {"findings": findings, "literature": lit, "synthesis": report}


if __name__ == "__main__":
    import io
    import requests as req
    from PIL import Image as PILImage

    url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
    img = PILImage.open(io.BytesIO(req.get(url, headers={"User-Agent": "test"}).content)).convert("RGB")
    out = run_pipeline([img], "Analyze this chest X-ray.", modality="Chest X-Ray")
    print("=== IMAGE DESCRIPTIONS ===\n", out["findings"])
    print("\n=== SYNTHESIS REPORT ===\n", out["synthesis"])
