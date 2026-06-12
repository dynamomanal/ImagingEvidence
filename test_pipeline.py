"""Run this to diagnose the analyze crash: python test_pipeline.py"""
import traceback
from dotenv import load_dotenv
load_dotenv()

print("=== Step 1: model imports ===")
try:
    from model import load_model, analyze_image
    print("OK")
except Exception as e:
    print("FAIL:", e); traceback.print_exc(); exit(1)

print("\n=== Step 2: agent imports ===")
try:
    from agent import run_medgemma, run_literature_search, run_synthesis
    print("OK")
except Exception as e:
    print("FAIL:", e); traceback.print_exc(); exit(1)

print("\n=== Step 3: load MedGemma model ===")
try:
    bundle = load_model()
    print("OK — device:", bundle["device"])
except Exception as e:
    print("FAIL:", e); traceback.print_exc(); exit(1)

print("\n=== Step 4: literature search ===")
try:
    lit = run_literature_search("reduced ejection fraction wall motion abnormality")
    print(f"OK — PubMed: {len(lit['pubmed'])}, PMC: {len(lit['pmc'])}")
except Exception as e:
    print("FAIL:", e); traceback.print_exc(); exit(1)

print("\n=== Step 5: Groq synthesis ===")
try:
    report = run_synthesis("Mildly reduced LVEF ~45%. Wall motion abnormality inferolateral.", lit)
    print("OK — first 200 chars:", report[:200])
except Exception as e:
    print("FAIL:", e); traceback.print_exc(); exit(1)

print("\n=== ALL STAGES PASSED ===")
