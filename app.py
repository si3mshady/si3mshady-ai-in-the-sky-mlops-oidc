# streamlit_app.py
import os, io, json, time, base64, traceback
import streamlit as st
import boto3
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

# --- Class list (fallback if endpoint doesn't return classes) ---
CLASSES = [
    "air_conditioner","car_horn","children_playing","dog_bark","drilling",
    "engine_idling","gun_shot","jackhammer","siren","street_music",
    "alarms","crowd","domestic","gunfire","police","grinding","forced_entry"
]

# --- Page setup ---
st.set_page_config(page_title="🎧 UrbanSound — Endpoint Tester", page_icon="🎧", layout="centered")
st.title("🎧 UrbanSound — SageMaker Endpoint Tester")
st.caption("Upload a clip → send to your SageMaker endpoint → see top prediction and probabilities.")

# --- Sidebar controls ---
DEFAULT_ENDPOINT = os.getenv("ENDPOINT_NAME", "urbansound-audio-staging")
DEFAULT_REGION   = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-2"))
DEFAULT_KEY      = os.getenv("PAYLOAD_KEY", "audio")

with st.sidebar:
    st.header("⚙️ Settings")
    endpoint = st.text_input("Endpoint name", value=DEFAULT_ENDPOINT)
    region   = st.text_input("AWS region",    value=DEFAULT_REGION)
    payload_fmt = st.radio("Payload format", ["JSON (base64)", "Binary (octet-stream)"], index=0)
    json_key = st.text_input("JSON key (when JSON)", value=DEFAULT_KEY)
    send_mode = st.selectbox("Preprocess before sending",
                             ["Raw file bytes", "Resampled mono 22.05 kHz WAV"],
                             index=1)
    topk = st.slider("Top-K to show", 3, min(10, len(CLASSES)), 7)
    st.markdown("**Class legend**")
    st.write(", ".join(CLASSES))

# --- Helpers ---
def to_resampled_wav_bytes(raw_bytes: bytes, target_sr: int = 22050) -> bytes:
    y, sr = librosa.load(io.BytesIO(raw_bytes), sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    buf = io.BytesIO()
    sf.write(buf, y, target_sr, format="WAV")
    buf.seek(0)
    return buf.read()

def show_mel(raw_bytes: bytes):
    try:
        y, sr = librosa.load(io.BytesIO(raw_bytes), sr=None, mono=True)
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512, power=2.0)
        S_db = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots()
        ax.imshow(S_db, origin="lower", aspect="auto")
        ax.set_title("Log-Mel Spectrogram (preview)")
        ax.set_xlabel("Frames"); ax.set_ylabel("Mel bins")
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.info(f"Could not render spectrogram: {e}")

def plot_topk(classes, probs, k):
    idx = np.argsort(probs)[::-1][:k]
    names = [classes[i] if i < len(classes) else str(i) for i in idx]
    vals  = [float(probs[i]) for i in idx]
    fig = plt.figure()
    plt.barh(range(len(vals)), vals)
    plt.yticks(range(len(vals)), names)
    plt.gca().invert_yaxis()
    plt.xlabel("Probability"); plt.xlim(0, 1)
    st.pyplot(fig, clear_figure=True)

# --- UI: upload & preview ---
file = st.file_uploader("Upload audio", type=["wav","mp3","ogg","flac","m4a","aiff","aif"])
if file:
    raw = file.read()
    st.audio(raw)
    with st.expander("Preview spectrogram", expanded=False):
        show_mel(raw)
else:
    st.info("Waiting for an audio file…")

# --- Predict ---
if st.button("Predict 🚀", type="primary", disabled=(file is None)):
    try:
        body_bytes = raw if send_mode == "Raw file bytes" else to_resampled_wav_bytes(raw, 22050)
        rt = boto3.client("sagemaker-runtime", region_name=region or None)

        if payload_fmt.startswith("JSON"):
            payload = json.dumps({json_key: base64.b64encode(body_bytes).decode("ascii")}).encode("utf-8")
            content_type = "application/json"
        else:
            payload = body_bytes
            content_type = "application/octet-stream"

        with st.spinner("Calling endpoint…"):
            t0 = time.time()
            resp = rt.invoke_endpoint(EndpointName=endpoint, ContentType=content_type, Body=payload)
            latency_ms = int((time.time() - t0) * 1000)
            body = resp["Body"].read()

        st.caption(f"Latency: **{latency_ms} ms** • ContentType sent: `{content_type}`")

        parsed = None
        try:
            parsed = json.loads(body)
        except Exception:
            pass

        if isinstance(parsed, dict):
            # Prefer server-provided classes; fallback to our CLASSES
            srv_classes = parsed.get("classes") or parsed.get("labels") or parsed.get("class_names")
            classes = srv_classes if (isinstance(srv_classes, list) and len(srv_classes) > 0) else CLASSES
            probs = parsed.get("probs") or parsed.get("scores") or parsed.get("probabilities")
            label = parsed.get("label") or parsed.get("prediction")

            if probs is not None and isinstance(probs, (list, tuple)) and len(probs) > 0:
                probs = np.asarray(probs, dtype=float)
                top = int(np.argmax(probs))
                top_label = classes[top] if top < len(classes) else str(top)
                top_prob  = float(probs[top])
                st.success(f"Top-1: **{top_label}** ({top_prob:.1%})")
                with st.expander("Top probabilities", expanded=True):
                    plot_topk(classes, probs, k=min(topk, len(probs)))
                with st.expander("Raw JSON response", expanded=False):
                    st.json(parsed)
            elif label is not None:
                st.success(f"Prediction: **{label}**")
                with st.expander("Raw JSON response", expanded=False):
                    st.json(parsed)
            else:
                st.subheader("JSON response")
                st.json(parsed)
        else:
            # Not JSON → show text/bytes
            try:
                txt = body.decode("utf-8", errors="ignore")
                st.subheader("Raw text response")
                st.code(txt)
            except Exception:
                st.subheader("Raw bytes response")
                st.write(body)

    except Exception as e:
        st.error(f"Invoke failed: {e}")
        st.code(traceback.format_exc())

st.caption("Tip: set ENDPOINT_NAME, AWS_REGION, PAYLOAD_KEY env vars to prefill settings.")

