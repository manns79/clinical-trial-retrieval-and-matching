from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clinical_trial_matching.ingestion.manifest import (
    build_source_manifest,
    manifest_to_json_record,
    sha256_file,
)


class ManifestTest(unittest.TestCase):
    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "source.txt"
            path.write_text("clinical trials\n", encoding="utf-8")

            self.assertEqual(
                sha256_file(path),
                "268cf01a0d9a61c3dceaa17326237a5a455e95d5454100b7f9380216f50df638",
            )

    def test_build_source_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "topics.xml"
            path.write_text("<topics />\n", encoding="utf-8")

            manifest = build_source_manifest(
                name="trec_2021_topics",
                source_url="https://example.test/topics.xml",
                input_path=path,
                dataset="trec_clinical_trials",
                year=2021,
                parser="trec_topics_xml",
                metadata={"track": "clinical_trials"},
            )

        record = manifest_to_json_record(manifest)
        self.assertEqual(record["schema_version"], "1.0")
        self.assertEqual(record["name"], "trec_2021_topics")
        self.assertEqual(record["dataset"], "trec_clinical_trials")
        self.assertEqual(record["year"], 2021)
        self.assertEqual(record["bytes"], 11)
        self.assertEqual(record["parser"], "trec_topics_xml")
        self.assertEqual(record["metadata"], {"track": "clinical_trials"})
        self.assertRegex(record["sha256"], r"^[a-f0-9]{64}$")
        self.assertTrue(record["created_at_utc"].endswith("Z"))


if __name__ == "__main__":
    unittest.main()
