# ImagingEvidence: AI-Assisted Cardiac Imaging Analysis with Evidence-Based Reporting

**A Clinical Documentation for Radiologists and Cardiologists**
*Research Preview | MVP Phase 1 Complete | MVP Phase 2 In Development*

---

## Executive Summary

**ImagingEvidence** is a multimodal cardiac imaging AI platform that ingests real clinical imaging studies, performs structured image analysis using state-of-the-art vision language models, retrieves relevant peer-reviewed literature in real time, and synthesizes a complete evidence-based clinical report — all within seconds of image upload.

The system is designed to function as an intelligent second reader: it does not replace the radiologist or cardiologist, but it dramatically accelerates the interpretation workflow, surfaces directly relevant research citations, and flags urgent findings that demand immediate attention.

What has been built and validated in MVP Phase 1 represents a full, working clinical AI pipeline covering six cardiac imaging modalities. MVP Phase 2 extends this with a locally-deployable, privacy-preserving open-source vision model that can run on institutional hardware or a free cloud GPU (Google Colab), eliminating dependency on external API providers for sensitive patient data scenarios.

---

## Why This Matters: The Clinical Problem

Cardiac imaging is one of the most information-dense disciplines in medicine. A single echocardiography study can contain 20–40 image views. A cardiac MRI late gadolinium enhancement (LGE) sequence can carry 120+ frames across multiple planes. A cardiac CT dataset can exceed 500 slices.

The radiologist or cardiologist interpreting these studies must simultaneously:

- Identify structural abnormalities across all views
- Quantify chamber dimensions and ejection fractions
- Apply current guideline thresholds (ASE, ESC, ACC/AHA)
- Correlate findings with clinical history
- Consider a differential diagnosis
- Recommend next steps
- Document findings in a structured report

This process takes 30–90 minutes per complex study. In a high-volume cardiac imaging lab, the reporting burden is immense. Meanwhile, the relevant research literature — which should ideally inform interpretation — is largely inaccessible in real time: the radiologist cannot search PubMed while dictating.

ImagingEvidence solves this. It does not perform one task; it performs the entire workflow — image interpretation, literature search, and report synthesis — in a unified pipeline, and delivers the result in under 60 seconds.

---

## What Has Been Built: System Architecture

The system is structured as a three-stage pipeline, each stage independent and auditable.

```
┌─────────────────────────────────────────────────────────────────┐
│                 Streamlit Web Interface (frontend.py)            │
│                                                                   │
│  [Upload Image/DICOM/Video] → [Select Modality & View]         │
│       → [Analyze] → [Evidence Report] → [Follow-up Chat]       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
 ┌────────────────┐ ┌─────────────┐ ┌──────────────────────┐
 │  Stage 1       │ │  Stage 2    │ │  Stage 3             │
 │  Vision Model  │ │  Literature │ │  Evidence Synthesis  │
 │  (agent.py)    │ │  Search     │ │  (agent.py)          │
 │                │ │  (tools.py) │ │                      │
 │ • Image decode │ │ • PubMed    │ │ • RAG context build  │
 │ • Measurements │ │ • Europe PMC│ │ • 700–950 word report│
 │ • Observations │ │ • Semantic  │ │ • Inline citations   │
 │ • Urgent flags │ │   Scholar   │ │ • Differential Dx    │
 └────────────────┘ └─────────────┘ │ • Next steps         │
          │                │        └──────────────────────┘
          └────────────────┘                  │
                                              ▼
                           ┌──────────────────────────────────┐
                           │  Vector Store (vector_store.py)   │
                           │  • Sentence-transformer embeddings│
                           │  • TF-IDF fallback               │
                           │  • Cosine similarity retrieval   │
                           └──────────────────────────────────┘
```

### Stage 1 — Vision Analysis

The uploaded image (or extracted video frame / DICOM slice) is passed to a vision language model with a highly structured, modality-specific system prompt. The system prompt is not generic — it is tuned per modality:

- **Echocardiography:** The model is instructed to report LVEF (Simpson's biplane preferred), LV dimensions (diastolic and systolic diameter, wall thickness), LA volume index, RV size and function, pericardial effusion, valve morphology (aortic, mitral, tricuspid, pulmonary), Doppler-equivalent findings where visible, and any incidental findings. ASE 2015/2016 guideline thresholds are embedded in the prompt.

- **Cardiac MRI (CMR):** Structured to capture LV and RV volumes, ejection fractions, myocardial mass, late gadolinium enhancement pattern (ischemic vs. non-ischemic distribution), T2 signal, native T1/T2 mapping where present, and pericardial findings.

- **Cardiac CT (CCTA):** Coronary artery segmental analysis (17-segment AHA model), stenosis severity grading (RADS/CAD-RADS), calcium scoring interpretation, plaque characterization (calcified vs. non-calcified), and non-cardiac incidental findings.

- **Nuclear Imaging (SPECT/PET):** Perfusion defect localization and size, reversibility (fixed vs. reversible), ejection fraction and wall motion abnormalities, gated SPECT findings.

- **Chest X-Ray:** Cardiac silhouette, cardiothoracic ratio, pulmonary vascularity, pleural effusions, mediastinal contour, bony structures, and lung fields — all in the context of cardiac disease.

- **Coronary Angiography:** TIMI flow grading, stenosis location and severity per vessel (LAD, LCx, RCA, branches), dominant coronary system, collateral circulation, and procedural status.

The output of Stage 1 is a structured plain-text clinical summary of approximately 200–400 words, formatted for direct ingestion into Stage 3 synthesis.

**Urgent Flags:** Stage 1 simultaneously evaluates for critical findings that require immediate clinical action:
- LVEF < 30% (severe systolic dysfunction)
- Cardiac tamponade physiology
- Massive pericardial effusion
- TIMI 0 or TIMI 1 coronary flow (acute occlusion)
- Large reversible perfusion defect (active ischemia)
- New left bundle branch block pattern

When any urgent flag is triggered, the UI surfaces a highlighted warning panel before the report, alerting the reviewing clinician.

---

### Stage 2 — Real-Time Literature Retrieval

This stage builds the evidence base for the report. Immediately after Stage 1 analysis, the pipeline:

1. **Parses the Stage 1 output** for recognizable cardiac conditions using a pattern library (e.g., detecting "dilated cardiomyopathy", "HFrEF", "hypertrophic cardiomyopathy", "myocarditis", "aortic stenosis", etc.)
2. **Constructs modality-aware search queries** incorporating MeSH (Medical Subject Headings) terms to improve precision
3. **Queries three independent sources in parallel:**
   - **PubMed** (NCBI eutils API): Up to 15 papers, sorted by relevance, filtered to cardiac imaging studies
   - **Europe PMC**: Up to 8 open-access full-text papers, prioritized for free-text availability
   - **Semantic Scholar**: Up to 6 highly-cited papers, selected for citation impact
4. **Deduplicates** all retrieved papers by URL
5. **Ingests all papers** into the vector store with sentence-transformer embeddings (all-MiniLM-L6-v2 model, 384-dimensional semantic embeddings)

The vector store uses cosine similarity for semantic retrieval. When GPU is unavailable, it falls back to TF-IDF bag-of-words similarity — ensuring the system functions in any environment, including CPU-only machines.

**Paper cache persistence:** Retrieved papers and their embeddings are cached to disk (`.paper_cache.json` and `.paper_cache_embeddings.npy`) so repeat sessions on similar conditions do not require re-querying external APIs. The RAG status indicator in the UI shows the radiologist exactly how many papers are currently indexed ("RAG active · 28 papers indexed").

---

### Stage 3 — Evidence Synthesis and Report Generation

The top-12 most semantically relevant papers are retrieved from the vector store and assembled into a structured context block. This context — combined with the Stage 1 vision analysis — is fed to a large language model with a strict report format prompt.

The output follows the **OpenEvidence format**, adapted for cardiac imaging:

1. **Opening paragraph:** Single-sentence diagnosis with confidence framing
2. **Key imaging findings (6–8 bullet points):** Each finding cites one or more papers inline using numbered references [1], [2], etc.
3. **Differential diagnosis:** Two categories — (a) most likely, (b) important not to miss
4. **Recommended next steps:** Targeted additional history questions (what to ask the patient), additional imaging (e.g., stress testing, CMR for suspected HCM), lab work (e.g., BNP, troponin, genetic panel), and referral recommendations
5. **Numbered reference list:** Author, title, journal, year, and DOI for all cited papers

Target report length is 700–950 words — substantive but not overwhelming.

---

### Image Ingestion Capabilities

The system handles the full range of clinical imaging file formats:

| Format | Handling |
|--------|----------|
| JPEG / PNG | Direct Pillow decode, base64 encode to API |
| Animated GIF | Frame extraction at configurable interval |
| MP4 / AVI / MOV | OpenCV frame sampling, user-selectable interval |
| DICOM (.dcm) | pydicom decode, Hounsfield unit rescaling, windowing, multi-frame cine support |

DICOM support is clinically significant: it means the system can ingest images directly from PACS export workflows without format conversion. The DICOM reader applies proper Hounsfield unit windowing (RescaleSlope, RescaleIntercept, WindowCenter, WindowWidth) and handles multi-frame DICOM cine sequences (e.g., cardiac gated CT slices, cine MRI loops).

---

## MVP Phase 1 — Groq Vision Pipeline (Completed)

**Status: Fully functional and deployed**

MVP Phase 1 represents the first complete, end-to-end working version of ImagingEvidence. It uses **Groq's cloud inference API** for both vision analysis and text synthesis.

### Why Groq?

Groq operates custom LPU (Language Processing Unit) hardware that delivers extremely low inference latency for large language models. Where a typical GPU-based cloud API might take 15–45 seconds to process a vision query, Groq delivers responses in 2–8 seconds. For a clinical tool where radiologist attention is the bottleneck, this latency difference is clinically meaningful.

### Models Used in Phase 1

| Stage | Model | Parameters | Role |
|-------|-------|------------|------|
| Vision Analysis | LLaMA 4 Scout 17B (16E MoE) | 17B active / ~109B total | Multimodal image understanding |
| Vision Fallback | LLaMA 3.2 11B Vision Preview | 11B | Secondary vision model |
| Synthesis | LLaMA 3.3 70B Versatile | 70B | Report generation with RAG |

**LLaMA 4 Scout** is a mixture-of-experts (MoE) multimodal model from Meta. The 16E MoE architecture activates only 17B parameters per token despite having a much larger total parameter count, making it computationally efficient at inference while retaining strong visual reasoning capability. It was selected as the primary vision model for its ability to describe structural cardiac imaging features with clinical specificity.

**LLaMA 3.3 70B Versatile** is a dense 70B parameter instruction-tuned model used for synthesis. At 70B parameters, it has sufficient capacity to follow the complex multi-constraint report format (OpenEvidence structure + inline citations + differential + next steps) reliably, without format breakdowns or hallucinated citations.

### What Phase 1 Delivers to the Clinician

When a radiologist uploads a cardiac image in Phase 1:

1. The image is analyzed in **2–8 seconds** (Groq Vision)
2. Literature is retrieved from **3 medical databases simultaneously**
3. A **700–950 word evidence-based report** is generated with inline citations
4. **Urgent findings** are flagged before the report
5. The radiologist can **ask follow-up questions** in a chat interface, with answers grounded in the same retrieved literature

This is a complete, clinically useful workflow delivered in under 60 seconds from image upload to structured report.

### Phase 1 Limitation: Cloud Dependency

The Groq API requires an internet connection and API key. For institutions with strict data governance policies — particularly those subject to HIPAA, GDPR, or NHS DSP Toolkit — sending imaging data to an external cloud endpoint may require a Business Associate Agreement (BAA) or may be prohibited entirely for identifiable imaging data.

This is the primary motivation for MVP Phase 2.

---

## MVP Phase 2 — Open-Source Local Vision Model (In Development)

**Status: MedGemma integration prototype built; Google Colab deployment pathway defined**

MVP Phase 2 replaces the Groq Vision API call (Stage 1) with an open-source vision language model that runs entirely on local or institution-controlled infrastructure. The text synthesis stage (Stage 3) also moves to an open-source model.

### The Phase 2 Vision Model: MedGemma

**MedGemma 1.5 4B** is Google DeepMind's open-weight medical vision language model, built on the Gemma 2 architecture and fine-tuned on a curated corpus of medical imaging data including radiology images, pathology slides, and clinical notes. It is released under a permissive open-weight license for research use.

The `model.py` file in this repository implements the complete MedGemma inference pipeline:

- **4-bit NF4 quantization** via `bitsandbytes`: Reduces the 4B parameter model from ~8GB to ~3.5GB GPU VRAM, making it accessible on consumer GPUs (RTX 3060, T4 Colab GPU)
- **SSD disk spill offloading**: When GPU VRAM is insufficient, model layers are offloaded to CPU RAM or SSD — enabling inference even without a dedicated GPU
- **RAM preflight check**: Warns the user if < 4.5GB system RAM is available before attempting to load the model
- **Background process management**: Kills competing GPU processes to free VRAM before model load

### Running Phase 2 on Google Colab

Google Colab provides free access to NVIDIA T4 GPUs (16GB VRAM) and A100 GPUs (40GB VRAM, Colab Pro). The Phase 2 pipeline can run entirely on a free T4 instance:

```
Colab T4 GPU (16GB VRAM)
    ├── MedGemma 1.5 4B @ 4-bit NF4 ≈ 3.5GB VRAM  ← Stage 1 Vision
    ├── Sentence-transformer all-MiniLM-L6-v2       ← Embedding model
    └── Synthesis LLM (Mistral 7B or LLaMA 3.1 8B) ← Stage 3 Synthesis
```

The radiologist or researcher uploads the Colab notebook, provides imaging data, and receives an evidence-based report — entirely within Google's infrastructure, with no external API calls.

For institutions with a Google Workspace or Google Cloud agreement, this provides a HIPAA-compatible pathway (Google Cloud HIPAA BAA covers Google Colab Enterprise).

### Phase 2 Candidate Models

| Stage | Model | Size | VRAM (4-bit) | License |
|-------|-------|------|--------------|---------|
| Vision | MedGemma 1.5 4B | 4B | ~3.5GB | Google Open |
| Vision (alt) | LLaVA-Med 7B | 7B | ~5.5GB | MIT |
| Vision (alt) | BioViL-T | Encoder | ~1GB | MIT |
| Synthesis | Mistral 7B Instruct | 7B | ~5.5GB | Apache 2.0 |
| Synthesis (alt) | LLaMA 3.1 8B Instruct | 8B | ~6GB | Meta Llama |
| Synthesis (alt) | Phi-3.5 Mini | 3.8B | ~3GB | MIT |

### Phase 2 Architecture Difference

The only architectural change between Phase 1 and Phase 2 is the **model source** in Stage 1 and Stage 3. The vector store, literature retrieval, DICOM processing, UI, and session management are identical. This modularity was an intentional design decision: the pipeline stages are decoupled so that any vision model can be substituted without changing the rest of the system.

```
Phase 1: Image → [Groq API call] → Structured text
Phase 2: Image → [Local HuggingFace model.generate()] → Structured text
```

---

## Training Your Own Cardiac Vision Model: Advantages and Disadvantages

A natural question from a clinician or researcher is: rather than using a general-purpose vision model prompted with cardiac instructions, why not fine-tune a model specifically on cardiac imaging data? This section explains the tradeoffs honestly.

### What "Training" Means in This Context

There are three levels of model customization, each with different requirements:

1. **Prompt engineering** (no training): Write better system prompts. This is what Phase 1 does.
2. **Fine-tuning / LoRA**: Take a pre-trained model (e.g., MedGemma 4B) and continue training it on your cardiac imaging dataset with supervised labels. This adjusts model weights to specialize for your data.
3. **Training from scratch**: Build and train a model architecture on cardiac imaging data from the beginning. Extremely expensive; only feasible for large institutions or companies.

For ImagingEvidence, the relevant question is fine-tuning — specifically, whether to fine-tune MedGemma or a similar open-source VLM on your institution's labeled cardiac imaging data.

---

### Advantages of Fine-Tuning a Cardiac-Specific Model

**1. Domain specialization improves measurement accuracy**
General-purpose vision models like LLaMA 4 Vision have seen broad internet image data but limited high-fidelity cardiac echo loops or cardiac MRI sequences. A model fine-tuned on 50,000 labeled echocardiograms with LVEF ground truth from Simpson's biplane measurement will produce more accurate quantitative outputs than a prompted general model.

**2. Structured output compliance**
Fine-tuning on output format examples dramatically increases the reliability of structured output (e.g., always producing the same field names, always reporting in the same units). Prompt engineering alone often produces inconsistent output formats; fine-tuning enforces the format at the weight level.

**3. Rare finding recognition**
Conditions like cardiac amyloidosis, arrhythmogenic cardiomyopathy (ARVC), or Fabry disease have subtle imaging phenotypes that a general model may not reliably identify. A fine-tuned model trained with expert-labeled examples of these conditions learns the visual signatures.

**4. Reduced inference cost at scale**
A fine-tuned 4B parameter model can outperform a prompted 70B model on specialized tasks. Running a 4B model costs roughly 17× less in compute than a 70B model — critical if the system is processing thousands of studies per month.

**5. Privacy-preserving pipeline end-to-end**
Once fine-tuned on local data, the model runs entirely on institutional hardware. No imaging data ever leaves the institution. This directly addresses the HIPAA/GDPR concern without requiring any cloud provider agreements.

---

### Disadvantages of Fine-Tuning a Cardiac-Specific Model

**1. Labeled training data is extremely hard to obtain**
Fine-tuning a vision language model on cardiac imaging requires paired data: image + expert-written clinical interpretation. Collecting 10,000–100,000 such pairs from radiologist reports is logistically difficult, legally complex (de-identification requirements), and institutionally sensitive. Most institutions do not have this data in a format usable for ML training.

**2. High compute cost for fine-tuning**
Even with LoRA (Low-Rank Adaptation, which only trains ~0.1–1% of model parameters), fine-tuning a 4B VLM requires a GPU with at least 16–24GB VRAM running for 12–72 hours. A full fine-tuning run costs $50–500 on cloud GPU infrastructure per experiment. Multiple experiments (hyperparameter sweeps, ablations) multiply this cost.

**3. Catastrophic forgetting**
Fine-tuning on a narrow distribution risks reducing the model's performance on out-of-distribution inputs. A model fine-tuned only on echocardiography may perform worse on cardiac CT or MRI compared to the base model. Careful curriculum design and evaluation sets are required.

**4. Regulatory and validation burden**
A fine-tuned model used in clinical workflows requires validation on a held-out test set, performance characterization across patient demographics (age, sex, BMI, heart rate), and ideally prospective validation. In the US, this triggers FDA 510(k) or De Novo pathways for AI-assisted diagnosis. In the EU, this falls under MDR Class IIa/IIb. This adds months to years of regulatory work.

**5. Maintenance as imaging hardware evolves**
Ultrasound machines, MRI scanners, and CT protocols change over time. A model trained on images from 2020 GE Vivid systems may not generalize to images from 2025 Canon or Philips machines. Continuous retraining and validation pipelines are required — a significant ongoing engineering investment.

**6. Hallucination risk in quantitative reporting**
Language models are intrinsically generative: they produce likely-sounding text. In cardiac imaging, a hallucinated LVEF value of 55% versus a true value of 35% has life-threatening clinical consequences. Any fine-tuned model must be evaluated for quantitative hallucination rates before clinical deployment.

---

### Recommendation for ImagingEvidence

For the current MVP phases, **prompt engineering with a capable base model** (Phase 1) and **MedGemma with quantization** (Phase 2) represent the appropriate engineering tradeoffs: fast to deploy, no labeling cost, no regulatory exposure, and still clinically useful as a second-reader tool.

Fine-tuning becomes the right investment when:
- A dataset of ≥ 10,000 labeled studies is available under institutional data governance
- A validation radiologist team is available to assess model outputs
- The use case moves from "research preview" to "clinical decision support tool"
- Institutional IT infrastructure can host 16–24GB VRAM GPU servers

At that point, the Phase 2 MedGemma integration in this codebase provides the exact training infrastructure hooks needed (model loading, quantization, format) to begin fine-tuning experiments with minimal additional engineering.

---

## Feature Summary by Phase

| Feature | Phase 1 (Completed) | Phase 2 (In Development) |
|---------|---------------------|--------------------------|
| Vision model | Groq LLaMA 4 Scout (cloud) | MedGemma 1.5 4B (local/Colab) |
| Synthesis model | Groq LLaMA 3.3 70B (cloud) | Mistral 7B / LLaMA 3.1 8B (local) |
| Internet required | Yes (Groq API) | No (fully offline capable) |
| Inference latency | 2–8 seconds | 15–60 seconds (T4 Colab) |
| Data privacy | Groq BAA required | Fully local, no external calls |
| Setup complexity | API key only | HuggingFace token + GPU required |
| DICOM support | Yes | Yes (identical pipeline) |
| Literature search | PubMed + PMC + Semantic Scholar | Same (unchanged) |
| RAG pipeline | Yes (vector store) | Yes (unchanged) |
| Modalities supported | 6 | 6 (same) |
| Urgent flag system | Yes | Yes (same) |
| Follow-up chat | Yes | Yes (same) |
| Cost per analysis | ~$0.002–0.008 (Groq tokens) | Free (Colab T4) or hardware cost |

---

## Clinical Safeguards and Limitations

All outputs of ImagingEvidence are explicitly framed as requiring clinician verification. The system is a **research preview**, not an FDA-cleared or CE-marked clinical decision support tool. The following safeguards are built into every report:

1. The report header labels every output as requiring radiologist/cardiologist review
2. Urgent findings are surfaced prominently but the system does not issue orders or alert clinical staff directly
3. All cited papers are real PubMed records — the citation links are verifiable
4. The system cannot access patient demographics, prior studies, or EHR context — interpretations are based solely on the uploaded image

These limitations are by design for an MVP phase, and are standard for AI systems at this stage of development.

---

## Technology Stack Summary

| Layer | Technology | Purpose |
|-------|------------|---------|
| Web UI | Streamlit 1.35+ | Interface, file upload, chat |
| Image Processing | Pillow, OpenCV, pydicom | Multi-format decode, DICOM handling |
| Vision AI (Phase 1) | Groq API, LLaMA 4 Scout | Image interpretation |
| Vision AI (Phase 2) | HuggingFace, MedGemma 4B | Local image interpretation |
| Text Synthesis | Groq API, LLaMA 3.3 70B | Report generation |
| Embeddings | sentence-transformers | Semantic paper retrieval |
| Vector Store | NumPy + cosine similarity + TF-IDF | Paper indexing and search |
| Literature APIs | PubMed, Europe PMC, Semantic Scholar | Evidence retrieval |
| Deployment | Streamlit Cloud / Docker / Colab | Hosting options |
| CI/CD | GitHub Actions | Automated testing |

---

## Conclusion

ImagingEvidence Phase 1 delivers a complete, working AI pipeline for cardiac imaging interpretation with evidence-based reporting. In a single workflow, it replaces manual literature search, structures the interpretation process across six imaging modalities, and surfaces critical findings that demand immediate attention — all in under 60 seconds.

Phase 2 extends this with a fully offline, privacy-preserving pathway using open-source medical vision models, making the system deployable in HIPAA-sensitive institutional environments without any cloud dependency.

The architecture is modular by design: the vision model, synthesis model, and retrieval pipeline are independently swappable. This means the system can evolve — incorporating fine-tuned cardiac-specific models as training data becomes available — without redesigning the platform.

For radiologists and cardiologists, ImagingEvidence represents a meaningful productivity tool today, with a clear roadmap toward a fully validated clinical decision support platform.

---

*ImagingEvidence is a research preview. All AI-generated interpretations require verification by a qualified clinician before any clinical action is taken. The system is not FDA-cleared or CE-marked.*

*Repository: https://github.com/dynamomanal/ImagingEvidence*
