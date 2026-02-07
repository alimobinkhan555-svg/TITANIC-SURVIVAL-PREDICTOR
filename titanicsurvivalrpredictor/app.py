import streamlit as st
import pandas as pd
import pickle

# ------------------------------
# Load Model + Scaler
# ------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]

model, scaler = load_model()

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="🚢 Titanic Survival Predictor",
    page_icon="🌌",
    layout="wide",
)

# ------------------------------
# Futuristic Styling
# ------------------------------
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        font-family: 'Trebuchet MS', sans-serif;
    }
    .prediction-box {
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        font-size: 22px;
        margin: 15px 0px;
        background: rgba(255, 255, 255, 0.1);
        color: #ecf0f1;
        backdrop-filter: blur(10px);
        box-shadow: 0px 8px 20px rgba(0,0,0,0.4);
        transition: all 0.3s ease-in-out;
    }
    .prediction-box:hover {
        transform: scale(1.02);
        box-shadow: 0px 12px 28px rgba(0,0,0,0.6);
    }
    .stProgress > div > div {
        background-image: linear-gradient(to right, #00c6ff , #0072ff);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# App Title
# ------------------------------
st.title("🌌 Futuristic Titanic Survival Predictor")
st.markdown("#### Will you survive the Titanic disaster? Fill in your details below 👇")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("🧾 Passenger Features")

def user_input_features():
    Pclass = st.sidebar.radio("Passenger Class (Pclass)", [1, 2, 3])
    Sex = st.sidebar.selectbox("Sex", ["female", "male"])
    Age = st.sidebar.slider("Age", 0, 80, 25)
    SibSp = st.sidebar.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
    Parch = st.sidebar.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
    Fare = st.sidebar.slider("Fare", 0, 500, 30)
    Embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])

    # Encoding same as training
    sex_map = {"female": 0, "male": 1}
    embarked_map = {"C": 0, "Q": 1, "S": 2}

    return pd.DataFrame({
        "Pclass": [Pclass],
        "Sex": [sex_map[Sex]],
        "Age": [Age],
        "SibSp": [SibSp],
        "Parch": [Parch],
        "Fare": [Fare],
        "Embarked": [embarked_map[Embarked]],
    })

input_df = user_input_features()

# ------------------------------
# Display Passenger Input
# ------------------------------
st.subheader("📋 Passenger Input Features")
st.dataframe(input_df.style.highlight_max(axis=1, color="lightblue"))

# ------------------------------
# Prediction
# ------------------------------
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
prediction_proba = model.predict_proba(input_scaled)[0]

# ------------------------------
# Results Display
# ------------------------------
st.subheader("🔮 Prediction Result")

if prediction == 1:
    st.markdown(
        f"<div class='prediction-box'> ✅ Passenger <b>SURVIVED</b><br> 🟢 Probability: {prediction_proba[1]:.2f} </div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"<div class='prediction-box'> ❌ Passenger <b>DID NOT SURVIVE</b><br> 🔴 Probability of Survival: {prediction_proba[1]:.2f} </div>",
        unsafe_allow_html=True,
    )

# Progress Bar for Survival Probability
st.subheader("📊 Survival Probability")
st.progress(float(prediction_proba[1]))

# ------------------------------
# Probabilities Side-by-Side
# ------------------------------
col1, col2 = st.columns(2)
col1.metric("🪦 Did Not Survive (0)", f"{prediction_proba[0]*100:.1f}%")
col2.metric("💚 Survived (1)", f"{prediction_proba[1]*100:.1f}%")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("🚀 Built with Streamlit | Futuristic UI 🌌")
