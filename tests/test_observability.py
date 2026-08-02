from __future__ import annotations

import io
import json
import logging
import unittest

from clinical_trial_matching.observability import elapsed_ms, log_event, now_ms


class ObservabilityTest(unittest.TestCase):
    def test_elapsed_ms_is_non_negative(self) -> None:
        start = now_ms()
        self.assertGreaterEqual(elapsed_ms(start), 0)

    def test_log_event_writes_json_payload(self) -> None:
        stream = io.StringIO()
        logger = logging.getLogger("clinical_trial_matching.tests.observability")
        logger.handlers = []
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        log_event(logger, "search", fields={"result_count": 2})

        payload = json.loads(stream.getvalue())
        self.assertEqual(payload, {"event": "search", "result_count": 2})


if __name__ == "__main__":
    unittest.main()
