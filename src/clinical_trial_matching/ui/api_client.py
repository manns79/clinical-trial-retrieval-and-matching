from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ApiError(Exception):
    status_code: int
    message: str

    def __str__(self) -> str:
        return f"API error {self.status_code}: {self.message}"


def search_trials_api(
    *,
    api_base_url: str,
    query: str,
    top_k: int,
    snippet_chars: int = 240,
    retriever: str = "sqlite-fts5",
    client: Any = None,
) -> dict[str, Any]:
    payload = {
        "query": query,
        "top_k": top_k,
        "snippet_chars": snippet_chars,
        "retriever": retriever,
    }
    response = _post(
        api_base_url=api_base_url,
        path="/search",
        payload=payload,
        client=client,
    )
    return _json_or_raise(response)


def get_trial_api(
    *,
    api_base_url: str,
    nct_id: str,
    client: Any = None,
) -> dict[str, Any]:
    response = _get(api_base_url=api_base_url, path=f"/trial/{nct_id}", client=client)
    return _json_or_raise(response)


def get_api_health(*, api_base_url: str, client: Any = None) -> dict[str, Any]:
    response = _get(api_base_url=api_base_url, path="/metrics/health", client=client)
    return _json_or_raise(response)


def _post(*, api_base_url: str, path: str, payload: dict[str, Any], client: Any = None) -> Any:
    if client is None:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Install project dependencies with `python3 -m pip install -e .`.") from exc
        with httpx.Client(timeout=30.0) as http_client:
            return http_client.post(_url(api_base_url, path), json=payload)
    return client.post(_url(api_base_url, path), json=payload)


def _get(*, api_base_url: str, path: str, client: Any = None) -> Any:
    if client is None:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Install project dependencies with `python3 -m pip install -e .`.") from exc
        with httpx.Client(timeout=30.0) as http_client:
            return http_client.get(_url(api_base_url, path))
    return client.get(_url(api_base_url, path))


def _json_or_raise(response: Any) -> dict[str, Any]:
    if response.status_code >= 400:
        message = _error_message(response)
        raise ApiError(status_code=response.status_code, message=message)
    payload = response.json()
    if not isinstance(payload, dict):
        raise ApiError(status_code=response.status_code, message="API response was not a JSON object")
    return payload


def _error_message(response: Any) -> str:
    try:
        payload = response.json()
    except ValueError:
        return str(getattr(response, "text", "Unknown error"))
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if detail:
            return str(detail)
    return str(payload)


def _url(api_base_url: str, path: str) -> str:
    return f"{api_base_url.rstrip('/')}{path}"
