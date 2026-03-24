import os

import joblib
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Dyslexia Detector",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-1: #fff8f1;
        --bg-2: #ffecd8;
        --surface: #ffffff;
        --surface-soft: #fff3e6;
        --text: #2b211a;
        --muted: #6f5847;
        --brand-a: #ff8a2a;
        --brand-b: #f0651e;
        --brand-c: #d64d0f;
        --line: #ffd7b2;
        --ok: #0f9f6e;
        --warn: #c98700;
        --risk: #e24646;
        --shadow-soft: 0 8px 22px -18px rgba(122, 45, 15, 0.35);
    }

    html, body, [data-testid="stAppViewContainer"], .stApp, .main {
        background:
            radial-gradient(circle at 8% 8%, rgba(255, 145, 58, 0.20), transparent 30%),
            radial-gradient(circle at 92% 22%, rgba(240, 101, 30, 0.14), transparent 28%),
            linear-gradient(145deg, var(--bg-1) 0%, var(--bg-2) 100%) !important;
        color: var(--text) !important;
        font-family: "Manrope", sans-serif !important;
    }

    [data-testid="stHeader"] {
        background: linear-gradient(90deg, #7a2d0f 0%, #9d390f 100%) !important;
    }

    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1320px;
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 0.85rem 1rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-soft);
        backdrop-filter: blur(6px);
    }

    .brand {
        font-weight: 800;
        letter-spacing: 0.2px;
        font-size: 0.95rem;
        color: #7a2d0f;
    }

    .status-pill {
        border: 1px solid var(--line);
        background: var(--surface-soft);
        border-radius: 999px;
        padding: 0.3rem 0.65rem;
        font-size: 0.74rem;
        color: #92400e;
        font-weight: 700;
    }

    .hero {
        border: 1px solid var(--line);
        background:
            linear-gradient(155deg, rgba(255, 255, 255, 0.94), rgba(255, 245, 235, 0.88));
        border-radius: 30px;
        padding: 2.1rem 1.5rem 1.5rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 12px 26px -20px rgba(122, 45, 15, 0.28);
    }

    .hero-badge {
        display: inline-block;
        border: 1px solid var(--line);
        background: var(--surface-soft);
        color: #92400e;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 0.35rem 0.75rem;
        margin-bottom: 0.9rem;
    }

    .hero h1 {
        margin: 0;
        line-height: 1.08;
        font-weight: 800;
        font-size: clamp(2rem, 4.8vw, 3.4rem);
        letter-spacing: -0.6px;
        background: linear-gradient(120deg, var(--brand-a), var(--brand-c));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero p {
        margin: 0.7rem auto 0 auto;
        max-width: 770px;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.65;
    }

    .stats {
        margin-top: 1.15rem;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
    }

    .stat {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.78rem 0.7rem;
        text-align: left;
    }

    .stat strong {
        display: block;
        color: #8a340f;
        font-size: 0.94rem;
        font-weight: 800;
    }

    .stat span {
        display: block;
        margin-top: 0.2rem;
        color: #6f5847;
        font-size: 0.8rem;
    }

    .section-title {
        font-weight: 800;
        color: #7a2d0f;
        font-size: 1.08rem;
        margin: 0.2rem 0 0.75rem 0;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin-bottom: 1.05rem;
    }

    .feature {
        border: 1px solid var(--line);
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(255, 247, 238, 0.92));
        border-radius: 16px;
        padding: 0.95rem;
        box-shadow: 0 6px 18px -18px rgba(122, 45, 15, 0.4);
    }

    .feature h4 {
        margin: 0 0 0.35rem 0;
        color: #7a2d0f;
        font-size: 0.93rem;
    }

    .feature p {
        margin: 0;
        color: #6f5847;
        font-size: 0.82rem;
        line-height: 1.45;
    }

    .card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.15rem;
        box-shadow: 0 8px 22px -18px rgba(122, 45, 15, 0.26);
        margin-bottom: 1rem;
    }

    [data-testid="stFileUploader"] {
        border: 2px dashed #f3a868;
        border-radius: 20px;
        background: linear-gradient(180deg, #fff7ee, #fff0e0);
        padding: 0.9rem;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--brand-a);
        background: #fff3e3;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(145deg, #fff0df, #ffe2c3) !important;
        border: 1px solid #ffbe84 !important;
        border-radius: 14px !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #7a2d0f !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(90deg, #ff8a2a, #d64d0f) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 700 !important;
    }

    .confidence-container {
        margin: 0.8rem 0 0.9rem 0;
    }

    .confidence-bar {
        background-color: #ffe6cc;
        border-radius: 999px;
        height: 10px;
        overflow: hidden;
    }

    .confidence-fill {
        background: linear-gradient(90deg, var(--brand-a), var(--brand-c));
        height: 100%;
        width: 0%;
        border-radius: 999px;
        transition: width 0.45s ease;
    }

    .confidence-text {
        margin-top: 0.45rem;
        font-size: 0.82rem;
        color: #6f5847;
        font-weight: 700;
    }

    .result-normal {
        background: linear-gradient(145deg, #e9fcf5, #d6f7e9);
        border-left: 6px solid var(--ok);
        border-radius: 16px;
        padding: 1rem;
        margin-top: 0.85rem;
    }

    .result-corrected {
        background: linear-gradient(145deg, #fff8da, #ffeead);
        border-left: 6px solid var(--warn);
        border-radius: 16px;
        padding: 1rem;
        margin-top: 0.85rem;
    }

    .result-reversal {
        background: linear-gradient(145deg, #fff1f1, #ffe1e1);
        border-left: 6px solid var(--risk);
        border-radius: 16px;
        padding: 1rem;
        margin-top: 0.85rem;
    }

    .suggestion {
        background: var(--surface-soft);
        border: 1px solid #ffd9b8;
        border-left: 4px solid var(--brand-c);
        border-radius: 14px;
        padding: 1rem;
    }

    .stButton button {
        background: linear-gradient(90deg, var(--brand-a), var(--brand-c));
        border: none;
        color: white;
        border-radius: 999px;
        padding: 0.56rem 1.5rem;
        font-weight: 700;
    }

    .stButton button:hover {
        box-shadow: 0 8px 16px -12px rgba(214, 77, 15, 0.5);
        transform: translateY(-1px);
    }

    .empty-state {
        border: 1px solid var(--line);
        background: linear-gradient(180deg, #fff8ef, #fff2e4);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        color: #8a4b24;
        font-weight: 600;
        text-align: center;
        margin-top: 0.25rem;
    }

    .streamlit-expanderHeader {
        background: #fffdfb;
        border: 1px solid var(--line);
        border-radius: 14px;
        color: #7a2d0f;
        font-weight: 700;
    }

    footer {
        text-align: center;
        color: #8a6c54;
        font-size: 0.8rem;
        margin-top: 1.4rem;
    }

    hr {
        border-color: #ffd9b8;
        margin: 0.9rem 0;
    }

    @media (max-width: 860px) {
        .stats, .feature-grid {
            grid-template-columns: 1fr;
        }
        .hero {
            padding: 1.5rem 1rem 1rem 1rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_cached(model_path: str):
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def preprocess_image(image, img_size=64):
    gray = image.convert("L")
    resized = gray.resize((img_size, img_size))
    array = np.array(resized, dtype=np.float32)
    return array.flatten().reshape(1, -1)


def main():
    st.markdown(
        """
        <div class="topbar">
            <div class="brand">Dyslexia Detector Cloud</div>
            <div class="status-pill">Landing UI v2</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">AI-assisted screening workflow</div>
            <h1>Dyslexia Detection Platform</h1>
            <p>Professional, fast handwriting analysis with structured class predictions and practical guidance for follow-up support.</p>
            <div class="stats">
                <div class="stat"><strong>3-class output</strong><span>Corrected, Normal, Reversal</span></div>
                <div class="stat"><strong>Instant inference</strong><span>Upload and analyze in seconds</span></div>
                <div class="stat"><strong>Action guidance</strong><span>Context-aware recommendations</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-title">Platform highlights</div>
        <div class="feature-grid">
            <div class="feature">
                <h4>Preprocessing consistency</h4>
                <p>Every image is normalized and resized before inference to keep predictions stable.</p>
            </div>
            <div class="feature">
                <h4>Confidence visibility</h4>
                <p>Confidence metrics surface certainty levels so decisions are easier to interpret.</p>
            </div>
            <div class="feature">
                <h4>Intervention-ready output</h4>
                <p>Each class maps to recommendations for practice and specialist consultation.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("About this tool", expanded=False):
        st.markdown(
            """
            This model analyzes handwriting patterns including:
            - Letter shape consistency
            - Spacing patterns
            - Reversal behavior such as b/d and p/q

            This tool is educational and not a medical diagnosis.
            Consult a qualified professional for formal assessment.
            """
        )

    model_path = "dyslexia_model.pkl"
    model = load_model_cached(model_path)
    if model is None:
        st.error(
            "Model file not found. Ensure `dyslexia_model.pkl` is in the app directory."
        )
        st.stop()

    img_size = 64
    st.markdown('<div class="section-title">Upload handwriting sample</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear and well-lit handwriting image.",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="empty-state">
                Upload an image to start analysis.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("Could not decode image. Please upload a valid PNG/JPG file.")
            st.stop()

        col1, col2 = st.columns([1, 1], gap="medium")

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded sample", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Analysis result")
            with st.spinner("Analyzing handwriting..."):
                input_data = preprocess_image(image, img_size)
                prediction = model.predict(input_data)
                pred_idx = int(prediction[0])
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_data)[0]
                    confidence = float(np.max(probabilities) * 100.0)
                else:
                    confidence = 100.0
                labels = ["Corrected", "Normal", "Reversal"]
                if 0 <= pred_idx < len(labels):
                    result = labels[pred_idx]
                else:
                    result = "Normal"

                st.markdown(
                    f"""
                    <div class="confidence-container">
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence:.2f}%;"></div>
                        </div>
                        <div class="confidence-text">Confidence: {confidence:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if result == "Normal":
                    st.markdown(
                        """
                        <div class="result-normal">
                            <h3 style="margin:0; color:#0f9f6e;">Normal writing pattern</h3>
                            <p style="margin:0.45rem 0 0 0;">The sample appears consistent with typical writing patterns.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif result == "Corrected":
                    st.markdown(
                        """
                        <div class="result-corrected">
                            <h3 style="margin:0; color:#b7791f;">Corrected writing pattern</h3>
                            <p style="margin:0.45rem 0 0 0;">The sample includes correction behavior that may indicate hesitation.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="result-reversal">
                            <h3 style="margin:0; color:#e24646;">Reversal pattern detected</h3>
                            <p style="margin:0.45rem 0 0 0;">Possible letter reversal behavior is present in this sample.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Suggestions")
        if result == "Reversal":
            st.markdown(
                """
                <div class="suggestion">
                    <ul style="margin:0; padding-left:1.2rem;">
                        <li><strong>Practice tracing letters</strong> focused on b/d and p/q.</li>
                        <li><strong>Use multi-sensory writing</strong> with tactile materials for muscle memory.</li>
                        <li><strong>Add color-coded cues</strong> for commonly reversed letters.</li>
                        <li><strong>Consult a specialist</strong> if this pattern is persistent.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif result == "Corrected":
            st.markdown(
                """
                <div class="suggestion">
                    <ul style="margin:0; padding-left:1.2rem;">
                        <li><strong>Encourage slow writing rhythm</strong> to improve confidence.</li>
                        <li><strong>Reinforce positive corrections</strong> without adding pressure.</li>
                        <li><strong>Use lined paper templates</strong> for stable spacing and size.</li>
                        <li><strong>Consider occupational support</strong> if frustration increases.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="suggestion">
                    <ul style="margin:0; padding-left:1.2rem;">
                        <li><strong>Maintain regular practice</strong> with short daily writing sessions.</li>
                        <li><strong>Use creative prompts</strong> to keep handwriting tasks engaging.</li>
                        <li><strong>Try different writing tools</strong> for comfort and control.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <footer>
            <hr>
            <p>This platform is informational and not a substitute for professional diagnosis or treatment.</p>
            <p>Powered by deep learning</p>
        </footer>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
