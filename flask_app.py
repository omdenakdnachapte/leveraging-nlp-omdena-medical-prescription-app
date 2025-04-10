from flask import Flask, render_template, request, redirect, url_for, flash
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import pickle
import os
import plotly.express as px
import json
import plotly

app = Flask(__name__)
app.secret_key = "your-secret-key"  # Required for flash messages

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model class (same as in Streamlit)
class OptimizedDiseaseClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(OptimizedDiseaseClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2048),
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

# Load model and artifacts
def load_model_and_artifacts(save_folder, num_classes):
    with open(os.path.join(save_folder, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    input_dim = len(tfidf.vocabulary_)

    model = OptimizedDiseaseClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    checkpoint = torch.load(os.path.join(save_folder, "best_model.pth"), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with open(os.path.join(save_folder, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    faiss_index = faiss.read_index(os.path.join(save_folder, "faiss_index.bin"))

    return model, tfidf, label_encoder, faiss_index

# Load dataset and encode Disease column
def load_dataset_and_symptoms(data_path, label_encoder):
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)
    df["Disease"] = label_encoder.transform(df["Disease"])
    all_symptoms = set()
    for symptoms in df["Symptoms"]:
        all_symptoms.update([s.strip() for s in symptoms.split(",")])
    return df, sorted(list(all_symptoms))

# Prediction function
def predict(symptoms, model, tfidf, label_encoder, faiss_index, df, confidence_threshold=0.6):
    symptoms_tfidf = tfidf.transform([symptoms]).toarray()
    symptoms_tensor = torch.tensor(symptoms_tfidf, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(symptoms_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    
    disease_name = label_encoder.inverse_transform([predicted_class])[0] if confidence >= confidence_threshold else "Uncertain"
    treatment = df[df["Disease"] == predicted_class].iloc[0].get("Treatment", "N/A") if confidence >= confidence_threshold else "N/A"

    symptoms_vec = symptoms_tfidf / (np.linalg.norm(symptoms_tfidf, keepdims=True) + 1e-6)
    distances, indices = faiss_index.search(symptoms_vec.astype(np.float32), 1)
    similar_disease = df.iloc[indices[0][0]]["Disease"]
    similar_disease_name = label_encoder.inverse_transform([similar_disease])[0]
    similarity_score = distances[0][0]

    top_k = 3
    top_probs, top_indices = torch.topk(probabilities, top_k)
    top_diseases = label_encoder.inverse_transform(top_indices.cpu().numpy())
    top_confidences = top_probs.cpu().numpy()

    return disease_name, treatment, similar_disease_name, confidence, top_diseases, top_confidences, similarity_score

# Global variables (loaded once)
save_folder = "models/model_classifier-002b-redo1"
data_path = "dataset/processed_diseases-priority.csv"
num_classes = len(pd.read_csv(data_path)["Disease"].unique())
model, tfidf, label_encoder, faiss_index = load_model_and_artifacts(save_folder, num_classes)
df_filtered, common_symptoms = load_dataset_and_symptoms(data_path, label_encoder)

# Routes
@app.route('/')
def index():
    return render_template('index.html', common_symptoms=common_symptoms, history=[])

@app.route('/predict', methods=['POST'])
def predict_route():
    symptoms = request.form.get('symptoms', '')
    selected_symptoms = request.form.getlist('common_symptoms')
    all_symptoms = ", ".join(selected_symptoms + [s.strip() for s in symptoms.split(",") if s.strip()])

    if not all_symptoms:
        flash("Please enter or select at least one symptom.", "error")
        return redirect(url_for('index'))

    try:
        disease, treatment, similar_disease, confidence, top_diseases, top_confidences, similarity = predict(
            all_symptoms, model, tfidf, label_encoder, faiss_index, df_filtered
        )

        # Generate Plotly chart
        fig = px.bar(
            x=top_diseases,
            y=top_confidences,
            labels={"x": "Disease", "y": "Confidence"},
            title="Top 3 Predicted Diseases",
            color=top_confidences,
            color_continuous_scale="Blues"
        )
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Simulate history (in Flask, you'd typically use a database or session)
        history = [{
            "Symptoms": all_symptoms,
            "Disease": disease,
            "Treatment": treatment,
            "Confidence": f"{confidence:.2%}",
            "Similar Disease": similar_disease,
            "Similarity": f"{similarity:.4f}"
        }]

        return render_template(
            'index.html',
            disease=disease,
            treatment=treatment,
            similar_disease=similar_disease,
            confidence=confidence,
            graph_json=graph_json,
            common_symptoms=common_symptoms,
            history=history
        )
    except Exception as e:
        flash(f"Prediction error: {e}", "error")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
