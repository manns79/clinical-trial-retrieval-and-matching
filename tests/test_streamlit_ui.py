from __future__ import annotations

import importlib.util
import unittest

HAS_STREAMLIT = importlib.util.find_spec("streamlit") is not None


@unittest.skipUnless(HAS_STREAMLIT, "Streamlit is not installed")
class StreamlitUiTest(unittest.TestCase):
    def test_available_retrievers_follow_api_health(self) -> None:
        from clinical_trial_matching.ui.streamlit_app import available_retrievers

        observed = available_retrievers(
            {
                "available_retrievers": [
                    "sqlite-fts5",
                    "fielded-bm25",
                    "dense",
                    "hybrid",
                    "unsupported",
                ]
            }
        )

        self.assertEqual(observed, ["sqlite-fts5", "fielded-bm25", "dense", "hybrid"])

    def test_retriever_label_is_human_readable(self) -> None:
        from clinical_trial_matching.ui.streamlit_app import retriever_label

        self.assertEqual(retriever_label("sqlite-fts5"), "SQLite FTS5")
        self.assertEqual(retriever_label("fielded-bm25"), "Fielded BM25")
        self.assertEqual(retriever_label("dense"), "Dense bi-encoder")
        self.assertEqual(retriever_label("hybrid"), "Hybrid RRF")


if __name__ == "__main__":
    unittest.main()
