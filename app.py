import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from groq import Groq
from fpdf import FPDF
from datetime import datetime
from utils.helper import DISPLAY_SYMPTOMS, build_input_vector, build_groq_prompt

st.set_page_config(
    page_title="Medical Symptom Analyzer",
    page_icon="🏥",
    layout="wide"
)

# ── Load model and data ──────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    le    = joblib.load("models/label_encoder.pkl")
    return model, le

@st.cache_data
def load_disease_info():
    desc  = pd.read_csv("data/symptom_Description.csv")
    prec  = pd.read_csv("data/symptom_precaution.csv")
    desc.columns  = desc.columns.str.strip()
    prec.columns  = prec.columns.str.strip()
    return desc, prec

model, le = load_model()
desc_df, prec_df = load_disease_info()

# ── Groq client ──────────────────────────────────────────────────────────────

def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not api_key:
        return None
    return Groq(api_key=api_key)

# ── Helper: get disease description & precautions ────────────────────────────

def get_disease_details(disease_name):
    desc_row = desc_df[desc_df["Disease"].str.strip().str.lower() == disease_name.lower()]
    prec_row = prec_df[prec_df["Disease"].str.strip().str.lower() == disease_name.lower()]
    description  = desc_row["Description"].values[0]  if not desc_row.empty  else ""
    prec_cols    = ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
    precautions  = ", ".join(
        [str(prec_row[c].values[0]) for c in prec_cols
         if not prec_row.empty and pd.notna(prec_row[c].values[0])]
    ) if not prec_row.empty else ""
    return description, precautions

# ── Helper: generate LLM advice ──────────────────────────────────────────────

def get_llm_advice(disease, symptoms, description, precautions):
    client = get_groq_client()
    if not client:
        return "⚠️ GROQ_API_KEY not set. Please add it in .streamlit/secrets.toml"
    prompt = build_groq_prompt(disease, symptoms, description, precautions)
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM Error: {str(e)}"

# ── Helper: generate PDF report ─────────────────────────────────────────────

def generate_pdf(selected_symptoms, predictions, llm_advice):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 100, 80)
    pdf.cell(0, 12, "Medical Symptom Analysis Report", ln=True, align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Generated on: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", ln=True, align="C")
    pdf.cell(0, 6, "DISCLAIMER: This is an AI-generated report. Not a substitute for medical advice.", ln=True, align="C")
    pdf.ln(6)

    pdf.set_fill_color(230, 245, 238)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(15, 110, 86)
    pdf.cell(0, 9, "Reported Symptoms", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 7, ", ".join(selected_symptoms))
    pdf.ln(4)

    pdf.set_fill_color(230, 245, 238)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(15, 110, 86)
    pdf.cell(0, 9, "Top Predicted Conditions", ln=True, fill=True)
    for i, (disease, prob) in enumerate(predictions, 1):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 7, f"{i}. {disease}  ({prob:.1f}% confidence)", ln=True)
    pdf.ln(4)

    pdf.set_fill_color(230, 245, 238)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(15, 110, 86)
    pdf.cell(0, 9, "AI-Generated Medical Advice", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    clean_advice = llm_advice.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 6, clean_advice)

    return pdf.output()

# ── UI ───────────────────────────────────────────────────────────────────────

st.title("🏥 Medical Symptom Analyzer")
st.markdown("*AI-powered disease prediction with personalized health advice*")
st.markdown("---")

col1, col2 = st.columns([1, 1.6], gap="large")

with col1:
    st.subheader("Select Your Symptoms")
    st.caption("Choose all symptoms you are currently experiencing")

    selected_symptoms = st.multiselect(
        "Search and select symptoms:",
        options=DISPLAY_SYMPTOMS,
        placeholder="Type to search symptoms..."
    )

    if selected_symptoms:
        st.info(f"**{len(selected_symptoms)}** symptom(s) selected")

    analyze_btn = st.button("🔍 Analyze Symptoms", type="primary",
                            disabled=len(selected_symptoms) == 0,
                            use_container_width=True)

    st.markdown("---")
    st.markdown("**⚠️ Important Notice**")
    st.caption(
        "This tool uses Machine Learning for educational purposes only. "
        "Always consult a qualified doctor for medical diagnosis and treatment."
    )

with col2:
    if analyze_btn and selected_symptoms:
        with st.spinner("Analyzing symptoms..."):
            input_vec  = build_input_vector(selected_symptoms)
            proba      = model.predict_proba(input_vec)[0]
            top3_idx   = np.argsort(proba)[::-1][:3]
            top3       = [(le.inverse_transform([i])[0], proba[i] * 100) for i in top3_idx]
            top_disease, top_prob = top3[0]
            description, precautions = get_disease_details(top_disease)

        st.subheader("Prediction Results")

        c1, c2, c3 = st.columns(3)
        for col, (dis, prob) in zip([c1, c2, c3], top3):
            col.metric(dis, f"{prob:.1f}%")

        fig = go.Figure(go.Bar(
            x=[p for _, p in top3],
            y=[d for d, _ in top3],
            orientation="h",
            marker_color=["#1D9E75", "#5DCAA5", "#9FE1CB"],
            text=[f"{p:.1f}%" for _, p in top3],
            textposition="auto"
        ))
        fig.update_layout(
            xaxis_title="Confidence (%)",
            yaxis_title="",
            height=220,
            margin=dict(l=0, r=0, t=10, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

        if description:
            with st.expander("📋 About this condition", expanded=True):
                st.write(description)

        st.markdown("---")
        st.subheader("🤖 AI Health Advice")
        with st.spinner("Generating personalized advice via Groq LLaMA-3..."):
            llm_advice = get_llm_advice(top_disease, selected_symptoms, description, precautions)
        st.markdown(llm_advice)

        st.markdown("---")
        st.subheader("📄 Download Report")
        pdf_bytes = generate_pdf(selected_symptoms, top3, llm_advice)
        st.download_button(
            label="⬇️ Download PDF Report",
            data=bytes(pdf_bytes),
            file_name=f"symptom_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    elif not analyze_btn:
        st.info("👈 Select your symptoms from the left panel and click **Analyze Symptoms**")
