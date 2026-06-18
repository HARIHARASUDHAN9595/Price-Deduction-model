import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# Load Model
model = joblib.load("model.pkl")

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.stButton>button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("💻 Laptop Price Predictor")
st.markdown("### Predict your laptop price instantly using AI")

st.divider()

# Two Columns
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox(
        "Brand",
        ["Apple","Dell","HP","Lenovo","Asus","Acer","MSI","Toshiba","Samsung","LG"]
    )

    typename = st.selectbox(
        "Laptop Type",
        ["Notebook","Ultrabook","Gaming","Workstation","2 in 1 Convertible","Netbook"]
    )

    inches = st.slider(
        "Screen Size (Inches)",
        10.0, 20.0, 15.6
    )

    ram = st.selectbox(
        "RAM (GB)",
        [2,4,8,16,32,64]
    )

with col2:
    screen = st.text_input(
        "Screen Resolution",
        "Full HD 1920x1080"
    )

    memory = st.text_input(
        "Storage",
        "256GB SSD"
    )

    gpu = st.text_input(
        "Graphics Card",
        "Intel HD Graphics 620"
    )

if company == "Apple":
    cpu = "Intel Core i5"
    opsys = "macOS"
else:
    cpu = st.text_input(
        "⚙ Processor",
        "Intel Core i5 7200U 2.5GHz"
    )

    opsys = st.selectbox(
        "🪟 Operating System",
        ["Windows 10","Windows 11","Linux","No OS"]
    )

st.divider()

if st.button("🔍 Predict Price"):
    data = pd.DataFrame({
        "Company": [company],
        "TypeName": [typename],
        "Inches": [inches],
        "ScreenResolution": [screen],
        "Cpu": [cpu],
        "Ram": [ram],
        "Memory": [memory],
        "Gpu": [gpu],
        "OpSys": [opsys]
    })

    prediction = model.predict(data)

    st.success(
        f"💰 Estimated Laptop Price: ₹{prediction[0]:,.2f}"
    )

    st.info(
        "Prediction generated using Machine Learning Model"
    )