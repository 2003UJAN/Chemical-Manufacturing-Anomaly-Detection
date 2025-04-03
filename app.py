import streamlit as st
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load trained models
iso_forest = joblib.load("models/isolation_forest.pkl")
scaler = joblib.load("models/scaler.pkl")
autoencoder = load_model("models/autoencoder.h5")

# Streamlit UI
st.set_page_config(page_title="Chemical Manufacturing Anomaly Detection", layout="wide")

st.title("🔬 Chemical Manufacturing Anomaly Detection")
st.markdown("Detect anomalies in **temperature, pressure, and chemical composition** in real-time.")

# Upload file option
uploaded_file = st.file_uploader("Upload sensor data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Scale data
    X_scaled = scaler.transform(df)

    # Isolation Forest Predictions
    anomaly_scores_if = iso_forest.decision_function(X_scaled)
    predictions_if = iso_forest.predict(X_scaled)
    predictions_if = [1 if x == -1 else 0 for x in predictions_if]

    # Autoencoder Predictions
    X_reconstructed = autoencoder.predict(X_scaled)
    reconstruction_error = np.mean(np.abs(X_scaled - X_reconstructed), axis=1)
    threshold = np.percentile(reconstruction_error, 95)
    predictions_ae = [1 if error > threshold else 0 for error in reconstruction_error]

    df["IsolationForest_Anomaly"] = predictions_if
    df["Autoencoder_Anomaly"] = predictions_ae

    # Show results
    st.write("### 📌 Anomaly Detection Results")
    st.dataframe(df)

    st.write("🔴 **Red = Anomaly detected**")
    st.markdown("#### ✅ Normal vs ❌ Anomalies")
    st.bar_chart(df["IsolationForest_Anomaly"].value_counts())

    # Save results
    df.to_csv("data/anomaly_results.csv", index=False)
    st.download_button("Download Results CSV", "data/anomaly_results.csv")


