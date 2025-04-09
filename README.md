## Leveraging NLP for Medical Prescription Administration
### Streamlit App Project by Omdena Kaduna Local Chapter Initiated by Jamaludeen Madaki Collaboratively executed by the Project Team

Overview This Streamlit-based application leverages Natural Language Processing (NLP) to predict diseases and recommend treatments based on user-input symptoms.

Developed by the Omdena Kaduna Local Chapter, the project aims to demonstrate how AI can assist in medical prescription administration by analyzing symptom data, providing disease predictions with confidence scores, and offering a lookup feature for disease-specific symptoms. 

The app integrates a PyTorch-trained neural network, TF-IDF vectorization, and FAISS for similarity search, making it a robust tool for educational exploration in healthcare.

Leveraging NLP for Medical Prescription Administration Natural Language Processing (NLP) plays a pivotal role in this project by transforming unstructured symptom descriptions into structured data for analysis. 
The app employs a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to convert textual symptom inputs into numerical representations, which are then processed by a custom-trained neural network (OptimizedDiseaseClassifier) to predict diseases. 
This approach enables: Symptom-to-Disease Mapping: Accurately identifying potential diseases from free-text or selected symptoms.

Confidence-Based Predictions: Providing probabilistic outputs to guide decision-making.

Scalability: The model can be extended with more data to cover additional diseases and symptoms.

Educational Value: Offering a lookup feature to explore symptoms associated with specific diseases, enhancing medical knowledge dissemination.

By bridging NLP and healthcare, this project showcases how AI can support medical professionals and communities in resource-limited settings, such as Kaduna, Nigeria, where access to immediate healthcare expertise may be constrained.

Features Symptom Input: Users can select common symptoms from a dropdown or enter custom symptoms in a text area.

Disease Prediction: Predicts diseases with confidence scores and suggests treatments based on a pre-trained model.

**Similar - Disease Lookup: Allows users to look up symptoms associated with specific diseases from a predefined dataset.

Prediction History: Tracks past predictions for reference.

Feedback System: Collects user feedback for continuous improvement.

Interactive Visualizations: Displays top predictions using Plotly bar charts.

Installation Prerequisites Python 3.8 or higher

Git (optional, for cloning the repository)

Setup Instructions Clone the Repository

`git clone https://github.com/Omdena-Kaduna/NLP-Medical-Prescription.git` 
`cd NLP-Medical-Prescription`

Create a Virtual Environment (optional but recommended)

`python -m venv venv source venv/bin/activate`  # On Windows: `venv\Scripts\activate`

Install Dependencies Install the required packages listed in requirements.txt:

`pip install -r requirements.txt`

Prepare Model and Data
Place the model artifacts (tfidf_vectorizer.pkl, best_model.pth, label_encoder.pkl, faiss_index.bin) in the model_classifier-002b folder.

Ensure the dataset file processed_diseases-priority.csv is in the root directory.

_Run the App _

`streamlit run app.py`

Open your browser and navigate to http://localhost:8501.

Requirements Create a file named requirements.txt with the following content:

streamlit==1.31.1 torch==2.3.0 pandas==2.2.2 numpy==1.26.4 scikit-learn==1.4.2 faiss-cpu==1.8.0 plotly==5.22.0

These versions are compatible as of April 2025; adjust them based on your environment or project updates. 

Usage Launch the App: Run the command above to start the Streamlit server.

Input Symptoms: Select symptoms from the dropdown or type them manually in the text area.

Predict: Click the "Predict" button to get disease predictions, treatments, and confidence scores.

Lookup Diseases: Use the "Lookup Symptoms by Disease" section to explore symptoms for a specific disease.

Review History: Check past predictions in the "Prediction History" section.

Provide Feedback: Share your experience via the feedback form.

Project Structure

NLP-Medical-Prescription/ │ ├── model_classifier-002b/ # Folder for model artifacts │ ├── tfidf_vectorizer.pkl │ ├── best_model.pth │ ├── label_encoder.pkl │ └── faiss_index.bin │ ├── processed_diseases-priority.csv # Dataset file ├── app.py # Main Streamlit application script ├── requirements.txt # Dependency list └── README.md # This file

Limitations Educational Purpose: This app is not a substitute for professional medical advice. 
**Always consult a healthcare provider for diagnosis and treatment**.

Dataset Dependency: Predictions and lookups rely on the accuracy and completeness of processed_diseases-priority.csv.

Model Scope: Limited to diseases and symptoms in the training data.

Contributing Contributions are welcome! To contribute: Fork the repository.

Create a feature branch (`git checkout -b` feature/YourFeature).

Commit your changes (`git commit -m` "Add your feature").

Push to the branch (`git push origin` feature/YourFeature).

Open a pull request.

Please adhere to the coding standards and include tests where applicable. 
Acknowledgments Omdena Kaduna Local Chapter: For fostering innovation and collaboration.

Jamaludeen Madaki: Project initiator and visionary.

Project Team: For their collaborative efforts in building this tool.

Open Source Community: For providing the tools (Streamlit, PyTorch, etc.) that made this possible.

License This project is licensed under the MIT License. See the LICENSE file for details. 
Contact For questions or support, reach out to the Omdena Kaduna Local Chapter via Omdena or contact Jamaludeen Madaki directly through the project repository.
