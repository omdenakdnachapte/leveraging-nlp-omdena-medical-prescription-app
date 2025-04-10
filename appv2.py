#removed faiss


# Streamlit app for modeling task 4
## @madakixo
# Purpose: Deploy a neural network to classify diseases based on symptoms, using TF-IDF features.
# Designed for interactive use with Streamlit and Google Drive integration.
# Enhanced with disease lookup functionality.

import streamlit as st  # Web app framework for interactive UI
import torch  # Core PyTorch library for tensor operations and neural networks
import torch.nn as nn  # Neural network module for defining layers and loss functions
import pandas as pd  # For data manipulation and DataFrame operations
import numpy as np  # For numerical computations and array handling
from sklearn.feature_extraction.text import TfidfVectorizer  # To convert symptom text into numerical features
import pickle  # To load Python objects (e.g., TF-IDF vectorizer, label encoder)
import os  # For file and directory operations
import plotly.express as px  # For interactive visualizations
import time  # For simulating processing time with progress bar

# Set page configuration
st.set_page_config(
    page_title="Leveraging NLP in Medical Prescription",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model class (must match training script)
class OptimizedDiseaseClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(OptimizedDiseaseClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2048),  # Increased capacity
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        self.dropout_layers = [layer for layer in self.network if isinstance(layer, nn.Dropout)]

    def forward(self, x):
        return self.network(x)

    def update_dropout(self, p):
        for layer in self.dropout_layers:
            layer.p = p

# Load model and artifacts from root directory (no FAISS)
@st.cache_resource
def load_model_and_artifacts(num_classes):
    try:
        with open("tfidf_vectorizer.pkl", "rb") as f:
            tfidf = pickle.load(f)
        input_dim = len(tfidf.vocabulary_)

        model = OptimizedDiseaseClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
        checkpoint = torch.load("best_model.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        return model, tfidf, label_encoder
    except Exception as e:
        st.error(f"Error loading model or artifacts: {e}")
        raise

# Load dataset and encode Disease column
@st.cache_data
def load_dataset_and_symptoms(data_path, _label_encoder):
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)
    df["Disease"] = _label_encoder.transform(df["Disease"])
    all_symptoms = set()
    for symptoms in df["Symptoms"]:
        all_symptoms.update([s.strip() for s in symptoms.split(",")])
    return df, sorted(list(all_symptoms))

# Prediction function without FAISS
def predict(symptoms, model, tfidf, label_encoder, df, confidence_threshold=0.6):
    symptoms_tfidf = tfidf.transform([symptoms]).toarray()
    symptoms_tensor = torch.tensor(symptoms_tfidf, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(symptoms_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    
    disease_name = label_encoder.inverse_transform([predicted_class])[0] if confidence >= confidence_threshold else "Uncertain"
    treatment = df[df["Disease"] == predicted_class].iloc[0].get("Treatment", "N/A") if confidence >= confidence_threshold else "N/A"

    top_k = 3
    top_probs, top_indices = torch.topk(probabilities, top_k)
    top_diseases = label_encoder.inverse_transform(top_indices.cpu().numpy())
    top_confidences = top_probs.cpu().numpy()

    return disease_name, treatment, confidence, top_diseases, top_confidences

# Custom CSS for larger font sizes
st.markdown(
    """
    <style>
    /* Increase font size globally */
    html, body, [class*="css"]  {
        font-size: 20px !important;
    }
    /* Increase header sizes */
    h1 {
        font-size: 40px !important;
    }
    h2 {
        font-size: 32px !important;
    }
    h3 {
        font-size: 28px !important;
    }
    /* Increase button text size */
    button {
        font-size: 20px !important;
    }
    /* Increase selectbox and text area font size */
    .stSelectbox, .stTextArea {
        font-size: 20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main app
def main():
    # Define paths (files are in the root directory)
    data_path = "processed_diseases-priority.csv"

    # Load data and model
    try:
        num_classes = len(pd.read_csv(data_path)["Disease"].unique())
        model, tfidf, label_encoder = load_model_and_artifacts(num_classes)
        df_filtered, common_symptoms = load_dataset_and_symptoms(data_path, label_encoder)
        st.success(f"Model and artifacts loaded successfully! Input dim: {len(tfidf.vocabulary_)}")
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return

    # Session state for prediction history and feedback
    if "history" not in st.session_state:
        st.session_state.history = []
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False

    # Sidebar navigation
    st.sidebar.title("🩺 Disease Predictor")
    st.sidebar.markdown("Select or type symptoms to predict diseases.")
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1053/1053171.png", width=100)

    with st.sidebar.expander("Advanced Options", expanded=False):
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Minimum confidence required for a prediction to be considered valid."
        )

    # Disease lookup section in sidebar
    with st.sidebar.expander("Lookup Symptoms by Disease", expanded=True):
        st.markdown("Select a disease to view its symptoms.")
        # List of diseases (unique and sorted)
        disease_list = sorted(set([
            "Vulvodynia", "Cold Sores", "Renal Cell Carcinoma", "Scabies", "Parkinson’s Disease",
            "Cleft Lip and Cleft Palate", "Peritoneal Cancer", "Sciatica", "Congenital Syphilis",
            "Colorectal Cancer", "Guillain-Barré Syndrome", "Noninsulin-Dependent Diabetes (Type 2)",
            "Pertussis", "Turner Syndrome", "Wilson’s Disease", "Atelectasis",
            "Malignant Pleural Mesothelioma", "Lung Cancer", "Prostate Cancer", "Kawasaki Disease",
            "Menopause", "Idiopathic Intracranial Hypertension", "Penile Cancer",
            "Disseminated Tuberculosis", "Bronchitis", "Giardiasis", "Dyslexia", "Factor V Leiden",
            "Dandruff", "Ramsay Hunt Syndrome", "Panic Disorder", "Bird Flu (Avian Influenza)",
            "Fifth Disease", "GERD", "Obstructive Sleep Apnea", "Colitis", "Neuroblastoma",
            "Sjögren’s Syndrome", "Bulimia", "Flu (Influenza)", "Anaphylaxis",
            "Left Ventricular Hypertrophy", "Piriformis Syndrome", "Hyponatremia",
            "Degenerative Disc Disease", "Ichthyosis Vulgaris", "Malaria", "Depression",
            "Pancreatic Cancer", "HIV/AIDS", "Myocarditis", "Aspergillosis", "Meningitis",
            "Renovascular Hypertension", "Aortic Stenosis", "Restless Legs Syndrome",
            "Schizoaffective Disorder", "COVID-19", "Down Syndrome", "Human Papillomavirus (HPV)",
            "Hyperhidrosis", "Pyloric Stenosis", "Hyperemesis Gravidarum", "Thrush (Oral Candidiasis)",
            "Thromboembolic Pulmonary Hypertension", "Cryptococcal Meningitis", "Hoarding Disorder",
            "Pancreatogenic Diabetes", "Raynaud’s Disease", "Hiccups (Chronic)",
            "Mitral Valve Prolapse", "Rotator Cuff Injury", "Ocular Rosacea", "Gallstones",
            "Cervical Dysplasia", "Congenital Zika Virus", "Spondylolisthesis", "Schistosomiasis",
            "Bubonic Plague", "Amyotrophic Lateral Sclerosis (ALS)", "Loiasis", "Interstitial Cystitis",
            "Dermatophytosis (Ringworm)", "Epididymitis", "Epiglottitis", "Male Infertility",
            "Lactose Intolerance", "Pink Eye (Conjunctivitis)", "Irritable Bowel Syndrome (IBS)",
            "Alcohol Poisoning", "Dry Socket", "Prurigo Nodularis", "Meningococcal Meningitis",
            "Pulmonary Edema", "Meningeal Leukemia", "Syphilis", "Anthrax", "Thyroid Cancer",
            "Athlete’s Foot", "Trigeminal Neuralgia", "Multiple Myeloma", "Food Allergy",
            "Lyme Disease", "Glaucoma", "Yaws", "Mpox (Monkeypox)", "High Cholesterol",
            "Urinary Tract Infection (UTI)", "Hiatal Hernia", "Heat Exhaustion", "Blepharitis",
            "Bronchiolitis", "Anorexia Nervosa", "Antisocial Personality Disorder", "Snoring",
            "Hemophilia", "Angina", "Gastroparesis", "Plantar Fasciitis",
            "Nephrogenic Diabetes Insipidus", "Cystic Fibrosis", "Liver Disease", "Bipolar Disorder",
            "Herpes Zoster Meningitis", "Teen Depression", "Scleritis", "Cytomegaloviral Pneumonia",
            "Geographic Tongue", "Cryosurgery for Prostate Cancer", "Tendinitis", "Morphea",
            "High Blood Pressure (Hypertension)", "Swine Influenza", "Compartment Syndrome",
            "Castleman Disease", "Non-Small Cell Lung Cancer", "Spinal Cord Injury",
            "Nasopharyngeal Carcinoma", "Heart Palpitations", "Coarctation of the Aorta",
            "Gastric Cancer", "Granuloma Annulare", "Fungal Meningitis", "Burkitt Lymphoma",
            "Nail Fungus", "Hemochromatosis", "Brief Psychotic Disorder", "Broken Arm",
            "Broken Wrist", "Still’s Disease", "Breast Cancer", "Amnesia",
            "Eosinophilic Esophagitis", "Anal Fistula", "Lupus Nephritis",
            "Enlarged Spleen (Splenomegaly)", "Vaginal Dryness", "Peyronie’s Disease", "Amblyopia",
            "Indigestion", "Erectile Dysfunction", "Type 3c Diabetes", "Acinetobacter Pneumonia",
            "Lipoma", "Dravet Syndrome", "African Trypanosomiasis", "Vasculitis", "Leukoplakia",
            "Lupus", "Patellofemoral Pain Syndrome", "Hypogonadism", "Exposure to Hepatitis B Virus",
            "Arm Pain", "Essential Tremor", "Pulmonary Arterial Hypertension", "Dermatitis",
            "Arrhythmia", "Onchocerciasis", "Marburg Virus Disease", "Nephroblastoma (Wilms Tumor)",
            "Tropical Sprue", "Small Cell Lung Cancer", "Buruli Ulcer", "Hairy Cell Leukemia",
            "Fallopian Tube Cancer", "Crohn’s Disease", "Yellow Fever", "Endometrial Cancer",
            "Sleep Apnea", "Zika Virus", "Anal Fissure", "Anaplastic Thyroid Cancer",
            "Cryptosporidiosis", "Overactive Bladder", "Stroke", "Autism Spectrum Disorder",
            "Neurosyphilis", "Chickenpox", "Tetanus", "Tinea Versicolor", "Leukemia", "Back Pain",
            "Tinnitus"
        ]))

        # Dropdown to select a disease
        selected_disease = st.selectbox("Choose a disease", options=disease_list, index=0)

        # Lookup and display symptoms
        if selected_disease:
            try:
                # Convert selected disease name to its encoded form using label_encoder
                encoded_disease = label_encoder.transform([selected_disease])[0]
                # Filter the dataset for the selected disease
                disease_row = df_filtered[df_filtered["Disease"] == encoded_disease].iloc[0]
                symptoms = disease_row["Symptoms"]
                st.write(f"**Symptoms for {selected_disease}:** {symptoms}")
                
                # Optional: Show additional info if available in the dataset
                if "Treatment" in disease_row:
                    st.write(f"**Treatment:** {disease_row['Treatment']}")
                else:
                    st.write("**Treatment:** Information not available in the dataset.")
            except Exception as e:
                st.warning(f"Could not find symptom data for {selected_disease}. Error: {e}")

    # Main content
    st.title("Interactive Disease Prediction")
    st.markdown("**Explore symptoms and get real-time disease predictions!**")

    # Symptom selection
    st.subheader("Select Symptoms")
    selected_symptoms = st.multiselect(
        "Choose common symptoms (or type below)",
        options=common_symptoms,
        help="Select from common symptoms or add custom ones below."
    )

    manual_symptoms = st.text_area(
        "Additional Symptoms (comma-separated)",
        placeholder="e.g., fatigue, headache",
        height=100,
        help="Enter additional symptoms not listed above, separated by commas."
    )

    symptoms = ", ".join(selected_symptoms + [s.strip() for s in manual_symptoms.split(",") if s.strip()])
    st.write(f"**Current Symptoms:** {symptoms if symptoms else 'None entered'}")

    if symptoms:
        st.info("Symptoms valid! Click 'Predict' to proceed.")
    else:
        st.warning("Please enter or select at least one symptom.")

    # Predict button with progress bar
    if st.button("Predict", key="predict_button"):
        if symptoms:
            progress_bar = st.progress(0)
            with st.spinner("Analyzing symptoms..."):
                for i in range(100):
                    time.sleep(0.01)  # Simulate processing time
                    progress_bar.progress(i + 1)
                try:
                    disease, treatment, confidence, top_diseases, top_confidences = predict(
                        symptoms, model, tfidf, label_encoder, df_filtered, confidence_threshold
                    )

                    # Display results with confidence badges
                    st.subheader("Prediction")
                    badge_color = "green" if confidence >= 0.75 else "orange" if confidence >= 0.6 else "red"
                    st.markdown(
                        f"**Disease:** {disease}  <span style='background-color:{badge_color};color:white;padding:2px 5px;border-radius:3px'>{confidence:.2%}</span>",
                        unsafe_allow_html=True
                    )
                    st.write(f"**Treatment:** {treatment}")

                    st.subheader("Top Predictions")
                    fig = px.bar(
                        x=top_diseases,
                        y=top_confidences,
                        labels={"x": "Disease", "y": "Confidence"},
                        title="Top 3 Predicted Diseases",
                        color=top_confidences,
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.session_state.history.append({
                        "Symptoms": symptoms,
                        "Disease": disease,
                        "Treatment": treatment,
                        "Confidence": f"{confidence:.2%}"
                    })

                except Exception as e:
                    st.error(f"Prediction error: {e}")
            progress_bar.empty()
        else:
            st.error("No symptoms provided!")

    # Prediction history
    if st.session_state.history:
        st.subheader("Prediction History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    # Feedback section
    st.subheader("Provide Feedback")
    st.write("Help us improve! Share your thoughts:")
    
    with st.expander("Quick Feedback", expanded=False):
        feedback_text = st.text_area("Your Feedback", placeholder="How was your experience? Any suggestions?")
        if st.button("Submit Feedback") and feedback_text:
            st.session_state.feedback_submitted = True
            st.success("Thank you for your feedback!")
        if st.session_state.feedback_submitted:
            st.write("Feedback submitted!")

    google_form_link = "https://forms.gle/your-google-form-link"  # Replace with your Google Form URL
    st.markdown(f"For detailed feedback, please fill out [this form]({google_form_link}).")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        **Note:** This app is for educational purposes only. Consult a healthcare professional for medical advice.
        Built with ❤️ using Streamlit, PyTorch, and Plotly
        Project initiated by Jamaludeen Madaki for Omdena Kaduna Impact Hub
        Collaboratively executed by the Project Team.
        """
    )

if __name__ == "__main__":
    main()
