from __future__ import annotations

import unittest
from typing import Any

from clinical_trial_matching.ui.api_client import ApiError, get_trial_api, search_trials_api


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self.payload = payload
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, Any] | None]] = []

    def post(self, url: str, json: dict[str, Any]) -> FakeResponse:
        self.requests.append(("POST", url, json))
        return FakeResponse(200, {"query": json["query"], "results": []})

    def get(self, url: str) -> FakeResponse:
        self.requests.append(("GET", url, None))
        return FakeResponse(200, {"nct_id": "NCT1", "title": "Trial"})


class UiApiClientTest(unittest.TestCase):
    def test_search_trials_api_posts_expected_payload(self) -> None:
        client = FakeClient()

        payload = search_trials_api(
            api_base_url="http://localhost:8000/",
            query="asthma",
            top_k=5,
            snippet_chars=120,
            client=client,
        )

        self.assertEqual(payload, {"query": "asthma", "results": []})
        self.assertEqual(
            client.requests,
            [
                (
                    "POST",
                    "http://localhost:8000/search",
                    {
                        "query": "asthma",
                        "top_k": 5,
                        "snippet_chars": 120,
                        "retriever": "fielded-bm25",
                    },
                )
            ],
        )

    def test_get_trial_api_gets_trial_detail(self) -> None:
        client = FakeClient()

        payload = get_trial_api(api_base_url="http://localhost:8000", nct_id="NCT1", client=client)

        self.assertEqual(payload["nct_id"], "NCT1")
        self.assertEqual(client.requests, [("GET", "http://localhost:8000/trial/NCT1", None)])

    def test_api_error_uses_detail_message(self) -> None:
        class ErrorClient:
            def get(self, url: str) -> FakeResponse:
                return FakeResponse(404, {"detail": "Trial not found"})

        with self.assertRaises(ApiError) as context:
            get_trial_api(api_base_url="http://localhost:8000", nct_id="NCT404", client=ErrorClient())

        self.assertEqual(context.exception.status_code, 404)
        self.assertEqual(context.exception.message, "Trial not found")


if __name__ == "__main__":
    unittest.main()
