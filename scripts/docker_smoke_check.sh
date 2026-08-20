#!/usr/bin/env sh
set -eu

IMAGE_NAME="${IMAGE_NAME:-clinical-trial-retrieval-and-matching:smoke}"
CONTAINER_NAME="${CONTAINER_NAME:-ctmatch-api-smoke}"
HOST_PORT="${HOST_PORT:-8000}"

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

on_exit() {
  status=$?
  trap - EXIT INT TERM
  if [ "$status" -ne 0 ] && docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    docker logs "$CONTAINER_NAME" >&2 || true
  fi
  cleanup
  exit "$status"
}

cleanup
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

docker build -t "$IMAGE_NAME" .

docker run \
  --detach \
  --name "$CONTAINER_NAME" \
  --publish "$HOST_PORT:8000" \
  --volume "$PWD/data/fixtures:/app/data/fixtures:ro" \
  --env TRIAL_CORPUS_PATH=/tmp/ctmatch/studies.sample.jsonl \
  --env TRIAL_STORE_PATH=/tmp/ctmatch/studies.sample_trial_store.sqlite \
  --env SQLITE_FTS_INDEX_PATH=/tmp/ctmatch/studies.sample_sqlite_fts5.sqlite \
  "$IMAGE_NAME" \
  sh -c '
    set -eu
    mkdir -p /tmp/ctmatch
    python -m clinical_trial_matching.cli ingest-ctgov-studies \
      --input /app/data/fixtures/ctgov_v2_studies.sample.json \
      --output /tmp/ctmatch/studies.sample.jsonl
    python -m clinical_trial_matching.cli build-sqlite-fts-index \
      --trials /tmp/ctmatch/studies.sample.jsonl \
      --output /tmp/ctmatch/studies.sample_sqlite_fts5.sqlite \
      --field-weight title=1.25 \
      --field-weight brief_summary=0.75 \
      --field-weight conditions=1.5 \
      --field-weight interventions=0.25 \
      --field-weight eligibility_criteria=0.5
    python -m clinical_trial_matching.cli build-trial-store \
      --trials /tmp/ctmatch/studies.sample.jsonl \
      --output /tmp/ctmatch/studies.sample_trial_store.sqlite
    exec uvicorn clinical_trial_matching.api.main:app \
      --host 0.0.0.0 \
      --port 8000
  '

for _ in $(seq 1 30); do
  if curl -fsS "http://localhost:$HOST_PORT/health" >/dev/null; then
    break
  fi
  sleep 2
done

curl -fsS "http://localhost:$HOST_PORT/health" >/dev/null

curl -fsS \
  -X POST "http://localhost:$HOST_PORT/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"adult persistent asthma inhaled corticosteroid","top_k":1}' \
  | python -c "import json, sys; payload=json.load(sys.stdin); assert payload['retriever']=='sqlite-fts5'; assert payload['results']; assert payload['results'][0]['nct_id']=='NCT99991001'"

curl -fsS "http://localhost:$HOST_PORT/trial/NCT99991001" \
  | python -c "import json, sys; payload=json.load(sys.stdin); assert payload['nct_id']=='NCT99991001'; assert payload['title']=='Synthetic Asthma Controller Therapy Study'"

echo "Docker smoke check passed."
