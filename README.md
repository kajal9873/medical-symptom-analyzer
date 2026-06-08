# 🏥 Medical Symptom Analyzer

An AI-powered web application that predicts diseases based on symptoms using Machine Learning and generates personalized health advice using Groq LLaMA-3.

## 🚀 Live Demo: https://medical-symptom-analyzer-zmwhi5sptjt64nvctuupzf.streamlit.app/

## 🧠 Tech Stack
| Component | Technology |
|-----------|------------|
| ML Model | Random Forest (scikit-learn) |
| LLM | Groq LLaMA-3 8B |
| Frontend | Streamlit |
| PDF Generation | fpdf2 |
| Visualization | Plotly |
| Deployment | Streamlit Cloud |

## 📊 Dataset
- **Source:** [Disease Symptom Description Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) — Kaggle
- **Size:** 4920 rows, 132 symptoms, 42 diseases
- **Format:** Binary symptom indicators + prognosis label

## ⚙️ Features
- 🔍 Select from 132 symptoms via searchable multi-select
- 🤖 Top-3 disease predictions with confidence scores
- 📊 Interactive probability bar chart
- 💬 Groq LLaMA-3 generated personalized health advice
- 📄 Downloadable PDF health report
- ⚡ Fast inference with cached ML model

## 🛠️ Local Setup

```bash
git clone https://github.com/kajal9873/medical-symptom-analyzer.git
cd medical-symptom-analyzer

pip install -r requirements.txt

# Add your Groq API key
echo 'GROQ_API_KEY = "your_key_here"' > .streamlit/secrets.toml

# Train the model first (run the notebook in Google Colab)
# Then place model.pkl and label_encoder.pkl in models/

streamlit run app.py
```

## 📁 Project Structure
```
medical-symptom-analyzer/
├── data/                   # Dataset CSV files
├── models/                 # Trained model + encoder
├── notebooks/              # EDA & training notebook
├── utils/
│   └── helper.py           # Symptom list, prompt builder
├── app.py                  # Main Streamlit app
├── requirements.txt
└── .streamlit/
    └── secrets.toml        # API keys (not committed)
```

## ⚠️ Disclaimer
This application is for educational purposes only. It is NOT a substitute for professional medical diagnosis or treatment. Always consult a qualified healthcare provider.
