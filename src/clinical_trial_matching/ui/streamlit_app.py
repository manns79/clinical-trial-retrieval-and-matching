from __future__ import annotations

import os
from typing import Any

from clinical_trial_matching.ui.api_client import (
    ApiError,
    get_api_health,
    get_trial_api,
    search_trials_api,
)

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - Streamlit entrypoint guidance
    raise RuntimeError("Install UI dependencies with `python3 -m pip install -e '.[ui]'`.") from exc


DEFAULT_API_BASE_URL = "http://localhost:8000"


def main() -> None:
    st.set_page_config(
        page_title="Clinical Trial Retrieval",
        page_icon="CT",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Clinical Trial Retrieval")
    st.caption("Research demo only. Not medical advice or an eligibility determination.")

    api_base_url = sidebar_controls()
    render_health(api_base_url)

    query = st.text_area(
        "Patient summary",
        value="Adult with persistent asthma and wheezing interested in an inhaled corticosteroid trial.",
        height=120,
    )

    col_top_k, col_button = st.columns([1, 3])
    with col_top_k:
        top_k = st.slider("Results", min_value=1, max_value=25, value=10)
    with col_button:
        st.write("")
        st.write("")
        submitted = st.button("Search", type="primary", use_container_width=False)

    if submitted:
        run_search(api_base_url=api_base_url, query=query, top_k=top_k)

    if "last_search" in st.session_state:
        render_search_results(st.session_state["last_search"])

    if "selected_nct_id" in st.session_state:
        st.divider()
        render_trial_detail(api_base_url, st.session_state["selected_nct_id"])


def sidebar_controls() -> str:
    with st.sidebar:
        st.header("Service")
        api_base_url = st.text_input(
            "API base URL",
            value=os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL),
        )
    return api_base_url


def render_health(api_base_url: str) -> None:
    try:
        health = get_api_health(api_base_url=api_base_url)
    except Exception as exc:
        st.warning(f"API unavailable: {exc}")
        return

    checks = health.get("checks", {})
    corpus_exists = checks.get("trial_corpus_exists")
    status_label = "ready" if corpus_exists else "corpus missing"
    st.info(
        f"API status: {health.get('status', 'unknown')} | "
        f"Corpus: {status_label} | "
        f"Path: {health.get('trial_corpus_path', 'unknown')}"
    )


def run_search(*, api_base_url: str, query: str, top_k: int) -> None:
    if not query.strip():
        st.error("Enter a patient summary before searching.")
        return
    try:
        payload = search_trials_api(api_base_url=api_base_url, query=query, top_k=top_k)
    except ApiError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        return

    st.session_state["last_search"] = payload


def render_search_results(payload: dict[str, Any]) -> None:
    results = payload.get("results", [])
    corpus = payload.get("corpus", {})
    st.subheader("Results")
    st.caption(
        f"{len(results)} shown from {corpus.get('trials', 0)} trials "
        f"({corpus.get('unique_nct_ids', 0)} unique NCT IDs)"
    )

    if not results:
        st.info("No matching trials returned.")
        return

    for result in results:
        render_result(result)


def render_result(result: dict[str, Any]) -> None:
    nct_id = str(result.get("nct_id", ""))
    title = str(result.get("title", "Untitled trial"))
    rank = result.get("rank", "")
    score = result.get("score", "")
    status = result.get("status", "")

    with st.container(border=True):
        left, right = st.columns([5, 1])
        with left:
            st.markdown(f"**{rank}. {title}**")
            st.caption(f"{nct_id} | {status} | score {score}")
        with right:
            if st.button("Details", key=f"details-{nct_id}", use_container_width=True):
                st.session_state["selected_nct_id"] = nct_id

        st.write(result.get("snippet", ""))
        matched_terms = result.get("matched_terms", [])
        if matched_terms:
            st.caption("Matched terms: " + ", ".join(str(term) for term in matched_terms))
        conditions = ", ".join(result.get("conditions", []))
        interventions = ", ".join(result.get("interventions", []))
        st.caption(f"Conditions: {conditions or 'none'}")
        st.caption(f"Interventions: {interventions or 'none'}")


def render_trial_detail(api_base_url: str, nct_id: str) -> None:
    try:
        trial = get_trial_api(api_base_url=api_base_url, nct_id=nct_id)
    except ApiError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Could not load trial detail: {exc}")
        return

    st.subheader(trial.get("title", "Trial Detail"))
    st.caption(f"{trial.get('nct_id', nct_id)} | {trial.get('status', '')}")

    cols = st.columns(3)
    cols[0].metric("Sex", trial.get("sex", "") or "Not specified")
    age = " - ".join(
        value for value in [trial.get("minimum_age", ""), trial.get("maximum_age", "")] if value
    )
    cols[1].metric("Age", age or "Not specified")
    cols[2].metric("Study Type", trial.get("study_type", "") or "Not specified")

    st.markdown("**Conditions**")
    st.write(", ".join(trial.get("conditions", [])) or "None listed")
    st.markdown("**Interventions**")
    st.write(", ".join(trial.get("interventions", [])) or "None listed")
    st.markdown("**Locations**")
    locations = trial.get("locations", [])
    st.write("\n".join(f"- {location}" for location in locations) or "None listed")
    st.markdown("**Eligibility Criteria**")
    st.text(trial.get("eligibility_criteria", "") or "None listed")


if __name__ == "__main__":
    main()
