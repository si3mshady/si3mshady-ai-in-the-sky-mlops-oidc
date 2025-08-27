#!/usr/bin/env bash
set -euo pipefail

JOB="${1:-${JOB:-}}"
REGION="${REGION:-us-east-2}"
EXPECTED_CLASSES="${EXPECTED_CLASSES:-17}"

if [[ -z "${JOB}" ]]; then
  echo "Usage: $0 <training-job-name>"
  echo "       REGION=us-east-2 EXPECTED_CLASSES=17 $0 <training-job-name>"
  exit 2
fi

say() { printf "%b\n" "$*"; }
hr() { printf "%0.s-" {1..80}; echo; }

# ---------- Describe training job ----------
say "🔎 Describing training job: ${JOB} (region=${REGION})"
DESC_JSON="$(aws sagemaker describe-training-job --region "$REGION" --training-job-name "$JOB")"

IMG="$(jq -r '.AlgorithmSpecification.TrainingImage' <<<"$DESC_JSON")"
ART="$(jq -r '.ModelArtifacts.S3ModelArtifacts' <<<"$DESC_JSON")"
START="$(jq -r '.TrainingStartTime' <<<"$DESC_JSON")"
END="$(jq -r '.TrainingEndTime' <<<"$DESC_JSON")"
INST="$(jq -r '.ResourceConfig.InstanceType' <<<"$DESC_JSON")"
HP="$(jq -r '.HyperParameters // {}' <<<"$DESC_JSON")"

say "$(hr)"
say "🧩 Training Image:"
say "  $IMG"
say
say "🕒 Timing:"
say "  Start: $START"
say "  End  : $END"
say
say "🖥️  Instance:"
say "  $INST"
say
say "⚙️  Hyperparameters (compact):"
jq -c '.' <<<"$HP"
say
say "📦 Model Artifact:"
say "  $ART"
say "$(hr)"

# ---------- Input data channels ----------
say "📂 Input data channels (S3):"
jq -r '.InputDataConfig[] | "- " + .ChannelName + " → " + .DataSource.S3DataSource.S3Uri' <<<"$DESC_JSON"
say "$(hr)"

# ---------- Download artifact to temp & inspect ----------
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

say "⬇️  Downloading artifact (temporary):"
aws s3 cp "$ART" "$TMP" --region "$REGION" >/dev/null
say "  saved to $TMP"
say

# helpers to handle .tar.gz or .tar
tlist() { tar -tzf "$TMP" 2>/dev/null || tar -tf "$TMP"; }
textract() {
  local path="$1"
  tar -xOzf "$TMP" "$path" 2>/dev/null || tar -xOf "$TMP" "$path"
}

# find classes.json in tar
CJSON="$(tlist | grep -i -m1 'classes\.json' || true)"

if [[ -z "$CJSON" ]]; then
  say "❌ No classes.json found inside artifact."
  exit 1
fi

say "🗂️  classes.json path in artifact:"
say "  $CJSON"
say

# count + list
COUNT="$(textract "$CJSON" | jq -r 'length')"
say "🔢 classes.json length: $COUNT  (expected: $EXPECTED_CLASSES)"
if [[ "$COUNT" != "$EXPECTED_CLASSES" ]]; then
  say "⚠️  MISMATCH: artifact has $COUNT classes; EXPECTED_CLASSES=$EXPECTED_CLASSES"
fi
say
say "📜 classes.json contents:"
textract "$CJSON" | jq -r '.[]' | nl -w2 -s": "
say "$(hr)"

# ---------- CloudWatch log tail ----------
say "🪵 CloudWatch log tail (last ~80 lines):"
LOGSTREAM="$(aws logs describe-log-streams --region "$REGION" \
  --log-group-name /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix "$JOB" \
  --order-by LastEventTime --descending --max-items 1 \
  --query 'logStreams[0].logStreamName' --output text 2>/dev/null || true)"

if [[ -n "$LOGSTREAM" && "$LOGSTREAM" != "None" ]]; then
  aws logs get-log-events --region "$REGION" \
    --log-group-name /aws/sagemaker/TrainingJobs \
    --log-stream-name "$LOGSTREAM" --limit 80 \
    --query 'events[].message' --output text
else
  say "  (no log stream found)"
fi

say "$(hr)"
say "✅ Done."

