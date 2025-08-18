
# streamlit_sagemaker_audio_client_flair.py
import os
import io
import json
import time
import base64
import traceback

import streamlit as st
import boto3

# Optional deps for visualization / resample
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

st.set_page_config(page_title="🎧 UrbanSound Endpoint Tester", page_icon="🎧", layout="centered")

# ---------- Styles (flair) ----------
st.markdown(
    '''
    <style>
      .result-card{
        border-radius: 18px;
        padding: 18px 20px;
        background: linear-gradient(135deg,#7c3aed 0%, #0ea5e9 45%, #10b981 100%);
        color: white;
        text-align: center;
        box-shadow: 0 8px 28px rgba(0,0,0,0.18);
      }
      .result-card .label{
        font-size: 28px;
        font-weight: 800;
        letter-spacing: 0.3px;
      }
      .result-card .prob{
        font-size: 18px;
        opacity: 0.95;
        margin-top: 6px;
      }
      .chip{
        display:inline-block;
        padding:6px 10px;
        background:#eef2ff;
        border-radius:999px;
        border:1px solid #e5e7eb;
        margin-right:8px;
        font-size:12px;
      }
    </style>
    ''',
    unsafe_allow_html=True
)

st.title("🎧 UrbanSound — SageMaker Endpoint Tester")
st.caption("Upload an audio file, hit **Predict**, and get a clean top‑1 label with probabilities.")

# --- Config ---
DEFAULT_ENDPOINT = os.getenv("ENDPOINT_NAME", "urbansound-audio-staging")
DEFAULT_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-2"))
DEFAULT_KEY = os.getenv("PAYLOAD_KEY", "audio")

with st.sidebar:
    st.header("⚙️ Settings")
    endpoint = st.text_input("Endpoint name", value=DEFAULT_ENDPOINT)
    region = st.text_input("AWS region", value=DEFAULT_REGION)
    payload_fmt = st.radio("Payload format", ["JSON (base64)", "Binary (octet-stream)"], index=0)
    json_key = st.text_input("JSON key (when JSON)", value=DEFAULT_KEY)
    send_mode = st.selectbox(
        "Preprocess before sending",
        ["Raw file bytes", "Resampled mono 22.05 kHz WAV"],
        index=1
    )
    st.caption("Match your serving container expectations.")

def _to_resampled_wav_bytes(raw_bytes: bytes, target_sr: int = 22050) -> bytes:
    y, sr = librosa.load(io.BytesIO(raw_bytes), sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    buf = io.BytesIO()
    sf.write(buf, y, target_sr, format="WAV")
    buf.seek(0)
    return buf.read()

def _show_mel(raw_bytes: bytes):
    try:
        y, sr = librosa.load(io.BytesIO(raw_bytes), sr=None, mono=True)
        y_plot = librosa.resample(y, orig_sr=sr, target_sr=22050) if sr != 22050 else y
        S = librosa.feature.melspectrogram(y=y_plot, sr=22050, n_mels=128, n_fft=2048, hop_length=512, power=2.0)
        S_db = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots()
        ax.imshow(S_db, origin="lower", aspect="auto")
        ax.set_title("Log‑Mel Spectrogram (preview)")
        ax.set_xlabel("Frames")
        ax.set_ylabel("Mel bins")
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.info(f"Could not render spectrogram: {e}")

def _plot_top_probs(classes, probs, k=5):
    try:
        import numpy as np
        idx = np.argsort(probs)[::-1][:k]
        names = [classes[i] if classes and i < len(classes) else str(i) for i in idx]
        vals = [probs[i] for i in idx]
        fig = plt.figure()
        plt.barh(range(len(vals)), vals)
        plt.yticks(range(len(vals)), names)
        plt.gca().invert_yaxis()
        plt.xlabel("Probability")
        plt.xlim(0, 1)
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.info(f"Could not plot probabilities: {e}")

file = st.file_uploader("Upload audio", type=["wav", "mp3", "ogg", "flac", "m4a", "aiff", "aif"])

if file is not None:
    raw = file.read()
    st.audio(raw)
    with st.expander("Preview features (optional)", expanded=False):
        _show_mel(raw)
else:
    st.info("Waiting for an audio file…")

if st.button("Predict 🚀", type="primary", disabled=file is None):
    try:
        to_send = raw if send_mode == "Raw file bytes" else _to_resampled_wav_bytes(raw, 22050)
        rt = boto3.client("sagemaker-runtime", region_name=region or None)

        if payload_fmt.startswith("JSON"):
            payload = json.dumps({json_key: base64.b64encode(to_send).decode("ascii")}).encode("utf-8")
            content_type = "application/json"
        else:
            payload = to_send
            content_type = "application/octet-stream"

        with st.spinner("Calling endpoint…"):
            t0 = time.time()
            resp = rt.invoke_endpoint(EndpointName=endpoint, ContentType=content_type, Body=payload)
            latency_ms = int((time.time() - t0) * 1000)
            body = resp["Body"].read()

        # Try parse JSON
        parsed = None
        try:
            parsed = json.loads(body)
        except Exception:
            pass

        st.caption(f"Latency: **{latency_ms} ms** • ContentType sent: `{content_type}`")

        if isinstance(parsed, dict):
            # Heuristics for common response shapes
            classes = parsed.get("classes") or parsed.get("labels") or parsed.get("class_names")
            probs = parsed.get("probs") or parsed.get("scores") or parsed.get("probabilities")
            label = parsed.get("label") or parsed.get("prediction")
            if probs is not None and classes is not None and len(classes) == len(probs):
                import numpy as np
                top = int(np.argmax(probs))
                top_label = classes[top]
                top_prob = float(probs[top])
                # Flair card
                st.markdown(f'<div class="result-card"><div class="label">{top_label}</div><div class="prob">{top_prob:.1%} confidence</div></div>', unsafe_allow_html=True)
                if top_prob >= 0.85:
                    st.balloons()
                with st.expander("Top probabilities", expanded=True):
                    _plot_top_probs(classes, probs, k=min(7, len(classes)))
                with st.expander("Raw JSON response", expanded=False):
                    st.json(parsed)
            elif label is not None:
                st.markdown(f'<div class="result-card"><div class="label">{label}</div><div class="prob">top‑1 prediction</div></div>', unsafe_allow_html=True)
                with st.expander("Raw JSON response", expanded=False):
                    st.json(parsed)
            else:
                st.subheader("JSON response")
                st.json(parsed)
        else:
            # Not JSON: try text
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

