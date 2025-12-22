import unittest
import sys
from pathlib import Path


# ✅ Add project root to PYTHONPATH so "import src.*" works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase4_generation import GenerationModule, GenerationConfig

def run_examples():
    config = GenerationConfig(
        top_k=5,
        retrieval_method="hybrid",
        rerank=False,
        strict_grounding=True,
        llm_backend="auto",
    )

    gen = GenerationModule(config=config)

    queries = [
        "educational toys for toddlers",
        "action figures superheroes",
        "building blocks construction set",
    ]

    for q in queries:
        out = gen.answer(q)
        print("=" * 80)
        print("QUERY:", q)
        print("-" * 80)
        print(out["response"])
        print("-" * 80)
        print("Sources:", [f"[{s['source_id']}] doc_id={s['doc_id']}" for s in out["sources"]])

if __name__ == "__main__":
    run_examples()
