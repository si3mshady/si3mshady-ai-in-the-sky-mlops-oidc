# streamlit_security_audio.py 
import os, io, json, time, base64, traceback
from typing import List, Optional

import streamlit as st
import boto3
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

# ---------- Config ----------
EXPECTED_CLASSES = ["gunfire", "glass_shatter"]  # ⬅️ updated to your two classes
DEFAULT_ENDPOINT = os.getenv("ENDPOINT_NAME", "urbansound-audio-staging")
DEFAULT_REGION   = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-2"))
DEFAULT_KEY      = os.getenv("PAYLOAD_KEY", "audio")

# Icons + severity coloring for a security tone
CLASS_STYLE = {
    "gunfire":       ("🔫", "critical"),
    "glass_shatter": ("🪟", "high"),
}
SEVERITY_BG = {"critical":"#fee2e2","high":"#fef3c7","medium":"#e0f2fe","info":"#ecfeff"}

# ---------- Page ----------
st.set_page_config(page_title="🔊 Security Audio Detector", page_icon="🔊", layout="centered")
st.title("🔊 Security Audio Detector — SageMaker")
st.caption("Upload an audio sample, we send it to your SageMaker endpoint, and visualize risk signals (gunfire vs glass_shatter).")

with st.sidebar:
    st.header("⚙️ Settings")
    endpoint = st.text_input("Endpoint name", value=DEFAULT_ENDPOINT)
    region   = st.text_input("AWS region", value=DEFAULT_REGION)
    payload_fmt = st.radio("Payload format", ["JSON (base64)", "Binary (octet-stream)"], index=0)
    json_key = st.text_input("JSON key (when JSON)", value=DEFAULT_KEY)
    send_mode = st.selectbox("Preprocess before sending",
                             ["Resampled mono 22.05 kHz WAV", "Raw file bytes"],
                             index=0)
    st.caption("Your container expects 22.05 kHz mono log-mels. Resampling recommended.")
    st.divider()
    st.markdown("**Expected classes**")
    st.write(", ".join(EXPECTED_CLASSES))

# ---------- Helpers ----------
def _resample_to_wav(raw: bytes, target_sr: int = 22050) -> bytes:
    y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    buf = io.BytesIO()
    sf.write(buf, y, target_sr, format="WAV")
    buf.seek(0)
    return buf.read()

def _show_mel(raw: bytes):
    try:
        y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
        S = librosa.feature.melspectrogram(y=y, sr=22050, n_mels=128, n_fft=2048, hop_length=512, power=2.0)
        S_db = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(7, 3))
        im = ax.imshow(S_db, origin="lower", aspect="auto")
        ax.set_title("Log-Mel Spectrogram")
        ax.set_xlabel("Frames")
        ax.set_ylabel("Mel bins")
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.info(f"Could not render spectrogram: {e}")

def _bar_plot(names: List[str], probs: List[float], top1: int):
    h = max(2, 0.6 * len(names))  # ensure all classes fit
    fig, ax = plt.subplots(figsize=(6.8, h))
    y = np.arange(len(names))
    ax.barh(y, probs)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    ax.set_xlim(0, 1)
    for i, p in enumerate(probs):
        ax.text(p + 0.01, i, f"{p:.2f}", va="center")
    st.pyplot(fig, clear_figure=True)

def _badge(label: str, prob: float):
    icon, sev = CLASS_STYLE.get(label, ("🔊", "info"))
    bg = SEVERITY_BG.get(sev, "#ecfeff")
    st.markdown(
        f"""
        <div style="background:{bg};padding:14px 16px;border-radius:14px;
                    border:1px solid rgba(0,0,0,0.06);">
          <div style="font-size:22px;font-weight:700;">
            {icon} Top-1: <span style="text-transform:none">{label}</span> ({prob:.1%})
          </div>
          <div style="opacity:0.8;margin-top:4px">Severity: <b>{sev.title()}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- App body ----------
file = st.file_uploader("Upload audio (wav/mp3/ogg/flac/m4a/aiff)", type=["wav","mp3","ogg","flac","m4a","aiff"])
if file:
    raw = file.read()
    st.audio(raw)
    with st.expander("Preview features", expanded=False):
        _show_mel(raw)
else:
    st.info("Waiting for an audio file…")

if st.button("Analyze 🔍", type="primary", disabled=not file):
    try:
        payload_bytes = _resample_to_wav(raw) if send_mode.startswith("Resampled") else raw
        content_type  = "application/json" if payload_fmt.startswith("JSON") else "application/octet-stream"
        body = (json.dumps({json_key: base64.b64encode(payload_bytes).decode("ascii")}).encode("utf-8")
                if content_type == "application/json" else payload_bytes)

        rt = boto3.client("sagemaker-runtime", region_name=region or None)
        with st.spinner("Calling endpoint…"):
            t0 = time.time()
            resp = rt.invoke_endpoint(EndpointName=endpoint, ContentType=content_type, Body=body)
            latency_ms = int((time.time() - t0) * 1000)
            payload = resp["Body"].read()

        # Parse response
        parsed: Optional[dict] = None
        try:
            parsed = json.loads(payload)
        except Exception:
            parsed = None

        st.caption(f"Latency: **{latency_ms} ms** • ContentType sent: `{content_type}`")

        if not isinstance(parsed, dict):
            st.error("Endpoint did not return JSON. See raw payload below.")
            st.code(payload[:1000])
        else:
            classes = parsed.get("classes") or parsed.get("labels") or EXPECTED_CLASSES
            probs   = parsed.get("probs")    or parsed.get("scores") or parsed.get("probabilities")

            # Safety: coerce numeric list
            if probs is not None:
                probs = [float(p) for p in probs]

            if probs and classes and len(probs) == len(classes):
                arr = np.array(probs)
                top = int(arr.argmax())
                _badge(classes[top], arr[top])

                with st.expander("Top probabilities", expanded=True):
                    _bar_plot(classes, list(arr), top)

                # Nudge if the model’s class list differs from expected set
                if set(c.lower() for c in classes) != set(EXPECTED_CLASSES):
                    st.info(f"The model’s class list differs from the {len(EXPECTED_CLASSES)} security classes shown in the sidebar. "
                            "If this is unexpected, retrain with the curated classes or confirm classes.json.")
                with st.expander("Raw JSON", expanded=False):
                    st.json(parsed)

            elif "label" in parsed:
                lbl = str(parsed["label"])
                _badge(lbl, float(parsed.get("confidence", 0.0)))
                with st.expander("Raw JSON", expanded=False):
                    st.json(parsed)
            else:
                st.warning("Could not find probabilities in the response. Showing raw JSON.")
                st.json(parsed)

    except Exception as e:
        st.error(f"Invoke failed: {e}")
        st.code(traceback.format_exc())

st.caption("Tip: set ENDPOINT_NAME, AWS_REGION, PAYLOAD_KEY environment variables to prefill settings.")
