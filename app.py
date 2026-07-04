#📌 Section 1 — Imports & Page Configuration

import streamlit as st
import numpy as np
import cv2
import pickle

from PIL import Image
from tensorflow.keras.models import load_model
#📌 Section 2 — Page Configuration

st.set_page_config(
    page_title="AI Hand Gesture Recognition",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

#📌 Section 3 — Load Model

@st.cache_resource
def load_ai_model():
    return load_model("model/gesture_model.keras")

model = load_ai_model()
#📌 Section 4 — Load Label Encoder

@st.cache_resource
def load_encoder():
    with open("model/label_encoder.pkl", "rb") as f:
        return pickle.load(f)

label_encoder = load_encoder()
#📌 Section 5 — Constants
IMG_SIZE = 128
#📌 Section 6 — Gesture Names & Descriptions



gesture_info = {

    "call_me": {
        "icon": "🤙",
        "title": "Call Me",
        "desc": "Gesture commonly used to indicate a phone call."
    },

    "fingers_crossed": {
        "icon": "🤞",
        "title": "Fingers Crossed",
        "desc": "Represents hope or wishing for good luck."
    },

    "okay": {
        "icon": "👌",
        "title": "Okay",
        "desc": "Indicates that everything is fine."
    },

    "paper": {
        "icon": "✋",
        "title": "Paper",
        "desc": "Open palm gesture."
    },

    "peace": {
        "icon": "✌",
        "title": "Peace",
        "desc": "Victory or peace sign."
    },

    "rock": {
        "icon": "✊",
        "title": "Rock",
        "desc": "Closed fist gesture."
    },

    "rock_on": {
        "icon": "🤘",
        "title": "Rock On",
        "desc": "Rock music hand sign."
    },

    "scissor": {
        "icon": "✌",
        "title": "Scissor",
        "desc": "Scissors gesture."
    },

    "thumbs": {
        "icon": "👍",
        "title": "Thumbs Up",
        "desc": "Approval or success."
    },

    "up": {
        "icon": "☝",
        "title": "Point Up",
        "desc": "Index finger pointing upward."
    }

}
#📌 Section 7 — Image Preprocessing

def preprocess(image):

    image = np.array(image)

    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    return image


st.markdown("""
<style>

.main-title{
    text-align:center;
    font-size:52px;
    font-weight:800;
    color:white;
}

.subtitle{
    text-align:center;
    color:#d6d6d6;
    font-size:20px;
    margin-bottom:30px;
}

.metric-card{
    padding:22px;
    border-radius:18px;
    text-align:center;
    color:white;
    box-shadow:0px 10px 25px rgba(0,0,0,0.25);
    transition:0.3s;
    margin-bottom:15px;
}

.metric-card:hover{
    transform:translateY(-5px);
}

.prediction-card{
    background:linear-gradient(135deg,#4facfe,#00f2fe);
    padding:25px;
    border-radius:20px;
    color:white;
    text-align:center;
    margin-top:25px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 class='main-title'>🖐 AI Hand Gesture Recognition System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Real-Time Hand Gesture Classification using Deep Learning</p>",
    unsafe_allow_html=True
)

st.write("")

c1, c2, c3, c4 = st.columns(4)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class='metric-card'
         style='background:linear-gradient(135deg,#667eea,#764ba2);'>
        <h3>📷 Images</h3>
        <h1>5,243</h1>
        <p>Training Samples</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class='metric-card'
         style='background:linear-gradient(135deg,#11998e,#38ef7d);'>
        <h3>🖐 Gestures</h3>
        <h1>10</h1>
        <p>Supported Classes</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class='metric-card'
         style='background:linear-gradient(135deg,#fc466b,#3f5efb);'>
        <h3>🧠 AI Model</h3>
        <h1>CNN</h1>
        <p>Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class='metric-card'
         style='background:linear-gradient(135deg,#f7971e,#ffd200);'>
        <h3>⚡ Framework</h3>
        <h1>TensorFlow</h1>
        <p>Computer Vision</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
left, right = st.columns([1.2, 1])

uploaded_file = left.file_uploader(
    "📤 Upload a Hand Gesture Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    # Read uploaded image
    image = Image.open(uploaded_file)

    # Display uploaded image
    left.image(
        image,
        caption="Uploaded Image",
        width=200
    )

    # ---------- PREPROCESS ----------
    img = preprocess(image)

    # ---------- PREDICT ----------
    prediction = model.predict(img, verbose=0)

    pred_index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    gesture = label_encoder.inverse_transform([pred_index])[0]

    info = gesture_info[gesture]

    # ---------- RESULT ----------
    with right:

        st.subheader("🎯 Prediction")

        st.markdown(f"# {info['icon']} {info['title']}")

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        st.info(info["desc"])

    # ---------- CONFIDENCE ----------
    st.markdown("### 🎯 Prediction Confidence")

    st.progress(confidence)

    # ---------- PROBABILITIES ----------
    st.markdown("---")

    st.markdown("## 📊 Prediction Probabilities")

    probabilities = prediction[0]

    for cls, prob in zip(label_encoder.classes_, probabilities):

        st.write(f"**{cls.replace('_',' ').title()}**")

        st.progress(float(prob))

        st.caption(f"{prob*100:.2f}%")

    st.markdown("---")
    st.success("Prediction completed successfully!")

st.markdown("## 🤖 About This Model")

st.info("""

**Model:** Custom CNN

**Framework:** TensorFlow/Keras

**Dataset:** 5,243 Images

**Classes:** 10

**Image Size:** 128×128

**Purpose:** Identify and classify hand gestures from uploaded images using a Convolutional Neural Network (CNN).

""")

st.markdown("---")

st.markdown(
"""
<center>


    Developed by Rashmi Deepak Toragallamath | TensorFlow • OpenCV • Streamlit


</center>
""",
unsafe_allow_html=True
)

st.markdown("---")
st.subheader("🤚 Supported Gestures")

c1, c2 = st.columns(2)

with c1:
    st.write("🤙 Call Me")
    st.write("🤞 Fingers Crossed")
    st.write("👌 Okay")
    st.write("✋ Paper")
    st.write("✌ Peace")

with c2:
    st.write("✊ Rock")
    st.write("🤘 Rock On")
    st.write("✂ Scissor")
    st.write("👍 Thumbs Up")
    st.write("☝ Point Up")
