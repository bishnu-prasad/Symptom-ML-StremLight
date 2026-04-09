import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Medical Diagnostic AI", page_icon="🩺", layout="centered")

# --- LOAD ML PIPELINE (Cached for speed) ---
# @st.cache_resource ensures the model only loads once when the server starts
# @st.cache_resource ensures the model only loads once when the server starts
@st.cache_resource
def load_pipeline():
    # Added "model/" to the file paths to match your folder structure
    with open("model/disease_prediction_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/symptom_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model/disease_label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("model/model_features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, encoder, features

try:
    model, scaler, encoder, features = load_pipeline()
    pipeline_loaded = True
except FileNotFoundError:
    st.error("🚨 Error: Could not find the .pkl files. Please ensure they are in the same folder as this script.")
    pipeline_loaded = False

# --- FRONTEND UI ---
st.title("🩺 Medical AI Diagnostic Assistant")
st.write("Select the patient's presenting symptoms below to generate a differential diagnosis based on our trained Logistic Regression model.")

if pipeline_loaded:
    # Clean up the feature names for the UI (e.g., 'chest_pain' -> 'Chest Pain')
    display_features = [f.replace('_', ' ').title() for f in features]
    
    # 1. User Input (Multiselect Box)
    selected_display_symptoms = st.multiselect(
        "Search and Select Symptoms:", 
        options=display_features,
        help="You can type to search for symptoms."
    )

    st.write("---")

    # 2. Prediction Logic
    if st.button("Generate Diagnosis", type="primary"):
        if not selected_display_symptoms:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            with st.spinner('Analyzing symptoms...'):
                # Map the nice UI names back to the exact machine learning feature names
                selected_features = [features[display_features.index(symp)] for symp in selected_display_symptoms]

                # Create the 45-column dataframe filled with 0s
                patient_df = pd.DataFrame(0, index=[0], columns=features)
                
                # Turn on the 1s for the symptoms the user selected
                for symp in selected_features:
                    patient_df.at[0, symp] = 1

                # MUST scale the data using our saved scaler
                patient_scaled = scaler.transform(patient_df)

                # Get the probabilities from the engine
                probabilities = model.predict_proba(patient_scaled)[0]

                # Find the Top 3 highest probabilities
                top_3_indices = np.argsort(probabilities)[::-1][:3]

                # --- DISPLAY RESULTS ---
                st.subheader("📋 Differential Diagnosis")
                
                for i, idx in enumerate(top_3_indices):
                    disease_name = encoder.inverse_transform([idx])[0].replace('_', ' ').title()
                    prob = probabilities[idx]
                    
                    # Formatting logic for visual severity
                    if i == 0 and prob > 0.60:
                        color = "green"
                    elif prob > 0.20:
                        color = "orange"
                    else:
                        color = "red"

                    st.markdown(f"#### {i+1}. {disease_name}")
                    st.write(f"**Confidence:** :{color}[{prob:.1%}]")
                    # Add a visual progress bar
                    st.progress(float(prob))

                # Clinical Safety Warning
                if probabilities[top_3_indices[0]] < 0.40:
                    st.info("💡 **Clinical Note:** The model has low confidence in a single primary diagnosis. This symptom presentation is highly atypical or overlaps heavily with multiple conditions. Comprehensive lab testing is recommended.")