import os
import keras_nlp
import keras

# Set backend
os.environ["KERAS_BACKEND"] = "tensorflow"

def load_model():
    print("⏳ Loading MyHealth Decoder (Gemma 2B)...")
    return keras_nlp.models.GemmaCausalLM.from_preset("gemma_2_2b_en")

def generate_explanation(model, medical_text):
    template = f"""Instruction: You are an empathetic and clear medical advocate. Your job is to explain complex medical reports to patients in simple, 5th-grade English.
    
    Medical Report:
    {medical_text}
    
    Explanation for Patient:
    """
    return model.generate(template, max_length=512)

if __name__ == "__main__":
    # Sample Radiology Report Snippet
    sample_report = """
    FINDINGS: There is a 5mm pulmonary nodule in the right upper lobe. No lymphadenopathy or pleural effusion. The cardiac silhouette is normal in size.
    IMPRESSION: Solitary pulmonary nodule, indeterminate. Follow-up CT in 6 months recommended.
    """
    
    try:
        model = load_model()
        print("\n🏥 Analyzing Report...")
        explanation = generate_explanation(model, sample_report)
        print("\n--- REPORT ---")
        print(sample_report.strip())
        print("\n--- PATIENT EXPLANATION ---")
        print(explanation)
        
    except Exception as e:
        print(f"❌ Error: {e}")
