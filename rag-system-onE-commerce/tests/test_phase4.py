import unittest
import sys
from pathlib import Path

# ✅ Add project root to PYTHONPATH so "import src.*" works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase4_generation import GenerationModule, GenerationConfig


class TestPhase4Generation(unittest.TestCase):
    def test_generation_runs(self):
        # Use fallback to avoid needing external API / heavy local models in tests
        config = GenerationConfig(
            top_k=3,
            retrieval_method="hybrid",
            rerank=False,
            strict_grounding=True,
            llm_backend="fallback",
        )

        gen = GenerationModule(config=config)
        out = gen.answer("educational toys for toddlers")

        self.assertIn("You asked:", out["response"])
        self.assertTrue(len(out["sources"]) <= 3)


if __name__ == "__main__":
    unittest.main()
