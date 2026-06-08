import json
import os
import numpy as np

# Load exact symptom list from Colab-generated JSON
_json_path = os.path.join(os.path.dirname(__file__), '..', 'symptom_list.json')
with open(_json_path, 'r') as f:
    SYMPTOMS_LIST = json.load(f)

DISPLAY_SYMPTOMS = [s.replace("_", " ").title() for s in SYMPTOMS_LIST]
SYMPTOM_MAP = {display: raw for display, raw in zip(DISPLAY_SYMPTOMS, SYMPTOMS_LIST)}


def build_input_vector(selected_display_symptoms):
    vector = np.zeros(len(SYMPTOMS_LIST))
    for display_sym in selected_display_symptoms:
        raw = SYMPTOM_MAP.get(display_sym)
        if raw and raw in SYMPTOMS_LIST:
            idx = SYMPTOMS_LIST.index(raw)
            vector[idx] = 1
    return vector.reshape(1, -1)


def build_groq_prompt(disease, symptoms, description="", precautions=""):
    symptom_text = ", ".join(symptoms) if symptoms else "various symptoms"
    prompt = f"""You are a helpful medical assistant AI. A patient has been diagnosed with a possible condition based on their symptoms.

Predicted Condition: {disease}
Reported Symptoms: {symptom_text}
{"Disease Description: " + description if description else ""}
{"Suggested Precautions: " + precautions if precautions else ""}

Please provide a clear, structured response with the following sections:
1. **About this condition** - Brief explanation in simple language (2-3 sentences)
2. **Why these symptoms occur** - Connect the symptoms to the condition (2-3 sentences)
3. **Immediate steps to take** - 3-4 actionable steps the patient can take right now
4. **When to see a doctor** - Specific warning signs that require urgent medical attention
5. **General precautions** - 3-4 lifestyle tips to manage or prevent worsening

Important: Always remind the user that this is an AI prediction tool and NOT a substitute for professional medical advice. Keep the language simple, empathetic and easy to understand.
"""
    return prompt
