"""
MyHealth Decoder — Patient Advocate App
========================================
Powered by MedGemma (HAI-DEF) for the Med-Gemma Impact Challenge.

Explains complex medical reports in simple, patient-friendly language.
Generates "Questions to Ask Your Doctor" based on findings.

Usage:
    # Full mode (requires GPU + MedGemma model weights):
    python app.py --mode full

    # Demo mode (uses pre-computed examples for development/testing):
    python app.py --mode demo
"""

import os
import argparse
import json

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_PRESET = "medgemma_instruct_4b"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are MyHealth Decoder, an empathetic and thorough medical advocate AI.
Your role is to help patients understand their medical reports.

Rules:
1. Explain ALL medical terms in simple, 5th-grade English.
2. Never diagnose or override the doctor's assessment.
3. Always recommend discussing findings with the treating physician.
4. Highlight what is NORMAL vs what needs ATTENTION.
5. Generate 3-5 specific questions the patient should ask their doctor.
6. Use a warm, reassuring but honest tone.
7. Structure your response clearly with headers.
"""

REPORT_PROMPT_TEMPLATE = """
<start_of_turn>user
{system_prompt}

Please analyze this medical report and explain it to me as a patient:

---
{report_text}
---

Provide:
1. **Simple Summary** — What does this report say in plain English?
2. **Key Findings** — Break down each finding (what's normal, what needs attention).
3. **What This Means For You** — Practical implications.
4. **Questions to Ask Your Doctor** — 3-5 specific, relevant questions.
<end_of_turn>
<start_of_turn>model
"""

# ─── Sample Reports for Demo Mode ────────────────────────────────────────────

SAMPLE_REPORTS = {
    "Chest X-Ray (Radiology)": """
CHEST X-RAY, PA AND LATERAL

CLINICAL INDICATION: Cough, shortness of breath.

FINDINGS:
The lungs are clear bilaterally. No focal consolidation, pleural effusion, or pneumothorax.
The cardiac silhouette is mildly enlarged with a cardiothoracic ratio of 0.55 (normal < 0.5).
The mediastinal contours are within normal limits.
There is mild degenerative change of the thoracic spine.
No acute osseous abnormality.

IMPRESSION:
1. Mild cardiomegaly. Recommend clinical correlation and echocardiogram if not recently performed.
2. No acute pulmonary disease.
3. Mild thoracic spondylosis.
""",
    "Blood Work (CBC + Metabolic)": """
COMPLETE BLOOD COUNT WITH DIFFERENTIAL

WBC: 11.2 x10^3/uL (H) [Reference: 4.5-11.0]
RBC: 4.85 x10^6/uL [Reference: 4.50-5.50]
Hemoglobin: 14.2 g/dL [Reference: 13.5-17.5]
Hematocrit: 42.1% [Reference: 38.0-50.0]
Platelets: 245 x10^3/uL [Reference: 150-400]

COMPREHENSIVE METABOLIC PANEL
Glucose: 126 mg/dL (H) [Reference: 70-100]
BUN: 18 mg/dL [Reference: 7-20]
Creatinine: 1.1 mg/dL [Reference: 0.7-1.3]
eGFR: 78 mL/min/1.73m2 [Reference: >60]
Sodium: 141 mEq/L [Reference: 136-145]
Potassium: 4.2 mEq/L [Reference: 3.5-5.0]
ALT: 45 U/L (H) [Reference: 7-35]
AST: 38 U/L (H) [Reference: 10-34]
HbA1c: 6.8% (H) [Reference: <5.7]
""",
    "Pathology Report": """
SURGICAL PATHOLOGY REPORT

SPECIMEN: Excisional biopsy, left breast, 2 o'clock position.

GROSS DESCRIPTION:
Received in formalin is a 3.2 x 2.1 x 1.8 cm tan-white, firm, irregular tissue fragment.
Serial sectioning reveals a 1.4 cm ill-defined, stellate, firm, white-tan mass.

MICROSCOPIC DESCRIPTION:
Sections show an invasive ductal carcinoma, grade 2 (of 3).
Nottingham score: 6/9 (tubule formation 2, nuclear pleomorphism 2, mitotic count 2).
Tumor size: 1.4 cm greatest dimension.
Margins: Negative, closest margin 0.3 cm (deep margin).
Lymphovascular invasion: Not identified.
Perineural invasion: Not identified.

IMMUNOHISTOCHEMISTRY:
ER: Positive (95%, strong)
PR: Positive (80%, moderate)  
HER2: Negative (score 1+)
Ki-67: 15%

IMPRESSION:
Invasive ductal carcinoma, grade 2, pT1c. ER+/PR+/HER2-. Margins negative.
""",
    "CT Scan (Pulmonary)": """
CT CHEST WITHOUT CONTRAST

CLINICAL INDICATION: Follow-up pulmonary nodule.

TECHNIQUE: Non-contrast helical CT of the chest with 1.25 mm reconstructions.

FINDINGS:
LUNGS: Previously identified 5mm nodule in the right upper lobe (series 4, image 62) 
is stable compared to prior exam dated 6 months ago. No new pulmonary nodules.
Mild dependent atelectasis noted bilaterally.

AIRWAYS: The trachea and main bronchi are patent. No endobronchial lesion.

PLEURA: No pleural effusion or thickening.

MEDIASTINUM: No significant lymphadenopathy. Heart size is normal.
No pericardial effusion.

BONES: Mild multilevel degenerative changes of the thoracic spine.
No suspicious osseous lesion.

IMPRESSION:
1. Stable 5mm right upper lobe pulmonary nodule. Given stability over 6 months,
   recommend follow-up CT in 12 months per Fleischner Society guidelines.
2. No new findings of concern.
"""
}

# Pre-computed demo explanations (for CPU-only mode)
DEMO_EXPLANATIONS = {
    "Chest X-Ray (Radiology)": """## 🩺 Simple Summary

Your chest X-ray shows that your **lungs look healthy** — no signs of pneumonia, fluid buildup, or collapsed lung. That's great news! However, the X-ray does show that your **heart appears slightly larger than normal**. This is called "mild cardiomegaly."

---

## 🔍 Key Findings

| Finding | What It Means | Status |
|---------|---------------|--------|
| **Lungs clear bilaterally** | Both lungs look normal, no infection or fluid | ✅ Normal |
| **No consolidation** | No areas where lung tissue is filled with fluid/pus | ✅ Normal |
| **No pleural effusion** | No fluid around the lungs | ✅ Normal |
| **No pneumothorax** | No collapsed lung | ✅ Normal |
| **Mild cardiomegaly (CTR 0.55)** | Heart is slightly bigger than expected (normal is under 0.50) | ⚠️ Needs Attention |
| **Thoracic spondylosis** | Normal wear and tear of the spine bones in your back | ℹ️ Age-related |

---

## 💡 What This Means For You

- Your **cough and shortness of breath** are likely **not** caused by a lung problem, since the lungs look clear.
- The **slightly enlarged heart** could be the reason for your shortness of breath. It might mean the heart is working harder than usual. This needs further testing (an **echocardiogram** — an ultrasound of the heart).
- The **spine changes** are common as we age and are usually not a concern.

---

## ❓ Questions to Ask Your Doctor

1. **"What could be causing my heart to be slightly enlarged?"** — There are many possible causes (high blood pressure, heart valve issues, etc.), and your doctor can help narrow it down.
2. **"Should I get an echocardiogram?"** — The report recommends one. Ask when this should happen.
3. **"Could the enlarged heart be causing my shortness of breath?"** — This will help connect your symptoms to the findings.
4. **"Is there anything I should do differently while waiting for more tests?"** — Ask about activity level, diet, and warning signs to watch for.
5. **"When should I follow up?"** — Make sure you have a clear plan for next steps.

---

> ⚕️ *Remember: This explanation is for your understanding only. Always discuss your results with your treating physician who has your full medical history.*
""",
    "Blood Work (CBC + Metabolic)": """## 🩺 Simple Summary

Your blood work shows mostly normal results, but there are a few values that are **slightly above normal** and deserve attention. The most important findings are an **elevated blood sugar level** (glucose and HbA1c), which suggests **diabetes** or **pre-diabetes**, and **mildly elevated liver enzymes**.

---

## 🔍 Key Findings

| Test | Your Value | Normal Range | Status |
|------|-----------|--------------|--------|
| **WBC (White Blood Cells)** | 11.2 | 4.5-11.0 | ⚠️ Slightly High |
| **RBC (Red Blood Cells)** | 4.85 | 4.50-5.50 | ✅ Normal |
| **Hemoglobin** | 14.2 | 13.5-17.5 | ✅ Normal |
| **Platelets** | 245 | 150-400 | ✅ Normal |
| **Glucose** | 126 | 70-100 | 🔴 High |
| **HbA1c** | 6.8% | <5.7% | 🔴 High (Diabetes range) |
| **Kidney (BUN/Creatinine)** | 18 / 1.1 | Normal ranges | ✅ Normal |
| **eGFR** | 78 | >60 | ✅ Normal |
| **Sodium/Potassium** | 141 / 4.2 | Normal ranges | ✅ Normal |
| **ALT (Liver)** | 45 | 7-35 | ⚠️ Mildly High |
| **AST (Liver)** | 38 | 10-34 | ⚠️ Mildly High |

---

## 💡 What This Means For You

- **HbA1c of 6.8%**: This is the most significant finding. An HbA1c between 5.7-6.4% is "pre-diabetes," and **6.5% or above indicates diabetes**. Your level of 6.8% falls in the diabetes range. This measures your average blood sugar over the past 2-3 months.
- **Glucose 126**: Confirms the elevated blood sugar. Fasting glucose above 126 on two occasions is diagnostic of diabetes.
- **Elevated liver enzymes (ALT/AST)**: These are only mildly elevated. Common causes include fatty liver, medications, or alcohol use. Not immediately alarming but should be monitored.
- **WBC slightly high**: Just barely above the upper limit. Could be a minor infection, stress, or normal variation. Not concerning in isolation.
- **Kidneys look good**: eGFR of 78 means your kidneys are functioning well.

---

## ❓ Questions to Ask Your Doctor

1. **"Do these results mean I have diabetes?"** — Your HbA1c and glucose both suggest this, but your doctor will confirm.
2. **"What lifestyle changes should I make right now?"** — Diet, exercise, and weight management can make a big difference.
3. **"Do I need medication for blood sugar control?"** — Depending on your situation, medication like metformin might be recommended.
4. **"What's causing my liver enzymes to be elevated?"** — Ask about fatty liver disease screening or if any of your medications could be the cause.
5. **"When should I retest?"** — You'll likely need follow-up blood work in 3 months.

---

> ⚕️ *Remember: This explanation is for your understanding only. Always discuss your results with your treating physician who has your full medical history.*
""",
    "Pathology Report": """## 🩺 Simple Summary

This pathology report confirms that the lump removed from your left breast is a type of **breast cancer** called **invasive ductal carcinoma**. I know that word "cancer" is scary, but this report actually contains several **encouraging signs**: the tumor is relatively small, the cancer cells have features that respond well to hormone treatment, and the surgeon was able to remove it completely with clear margins.

---

## 🔍 Key Findings

| Finding | What It Means | Status |
|---------|---------------|--------|
| **Invasive ductal carcinoma** | The most common type of breast cancer; started in milk ducts and spread into surrounding tissue | 🔴 Cancer confirmed |
| **Grade 2 of 3** | Moderately aggressive; not the most aggressive, not the least | ⚠️ Moderate |
| **Tumor size: 1.4 cm** | About the size of a small marble — considered relatively small | ✅ Favorable |
| **Margins: Negative** | The surgeon got all the cancer out; no cancer cells at the edges | ✅ Very Good |
| **Closest margin: 0.3 cm** | The nearest edge is 3mm away from cancer — sufficient clearance | ✅ Good |
| **No lymphovascular invasion** | Cancer has NOT spread into blood or lymph vessels | ✅ Very Good |
| **No perineural invasion** | Cancer has NOT spread along nerves | ✅ Very Good |
| **ER+ (95%)** | Cancer cells are very sensitive to estrogen — hormone therapy will work well | ✅ Good for Treatment |
| **PR+ (80%)** | Cancer cells also respond to progesterone | ✅ Good for Treatment |
| **HER2- (1+)** | Cancer does NOT have excess HER2 protein | ℹ️ Neutral |
| **Ki-67: 15%** | Shows how fast cancer cells are dividing — 15% is in the low-to-moderate range | ✅ Favorable |

---

## 💡 What This Means For You

- **The good news**: Your cancer was caught relatively early (stage T1c = small tumor), the surgeon removed it completely, and it's the type that responds very well to **hormone-blocking medication** (like tamoxifen or an aromatase inhibitor).
- **ER+/PR+/HER2-** is actually the **most common and most treatable** subtype of breast cancer. Hormone therapy is very effective for this type.
- **No lymphovascular invasion** means the cancer has not started spreading into blood vessels or lymph channels, which is a very positive sign.
- **Next steps** will likely include discussion about radiation therapy, hormone therapy, and possibly additional testing (like Oncotype DX) to determine if chemotherapy is needed.

---

## ❓ Questions to Ask Your Doctor

1. **"What stage is my cancer?"** — The pathology gives the T (tumor) classification, but your doctor will combine this with lymph node and imaging results for the full stage.
2. **"Will I need radiation therapy after surgery?"** — This is common after lumpectomy (partial breast removal).
3. **"Should I get an Oncotype DX test?"** — This genomic test helps predict whether chemotherapy would benefit you, especially for ER+/HER2- cancers.
4. **"What hormone therapy do you recommend, and for how long?"** — Given the strong ER/PR positivity, hormone therapy will likely be recommended for 5-10 years.
5. **"What is my prognosis?"** — With small tumor size, clear margins, and favorable receptor status, outcomes are generally very good.

---

> ⚕️ *Remember: This explanation is for your understanding only. Always discuss your results with your treating physician who has your full medical history.*
""",
    "CT Scan (Pulmonary)": """## 🩺 Simple Summary

Great news! Your follow-up CT scan shows that the **small spot (nodule) in your lung is the same size** as it was 6 months ago. When a nodule stays the same size over time, it's almost always **benign (not cancer)**. The rest of your chest looks healthy.

---

## 🔍 Key Findings

| Finding | What It Means | Status |
|---------|---------------|--------|
| **5mm right upper lobe nodule — STABLE** | The small spot hasn't grown in 6 months | ✅ Very Reassuring |
| **No new pulmonary nodules** | No new spots have appeared | ✅ Normal |
| **Mild dependent atelectasis** | Tiny areas of collapsed lung at the bottom (very common, from lying down during the scan) | ✅ Normal |
| **Airways patent** | Air passages are open and clear | ✅ Normal |
| **No pleural effusion** | No fluid around the lungs | ✅ Normal |
| **No lymphadenopathy** | Lymph nodes are normal size | ✅ Normal |
| **Heart size normal** | Heart looks healthy | ✅ Normal |
| **Degenerative spine changes** | Normal wear and tear of the spine | ℹ️ Age-related |

---

## 💡 What This Means For You

- **Stability = good news.** Cancer nodules almost always grow over 6 months. The fact that yours hasn't changed is very reassuring.
- The **Fleischner Society guidelines** are the standard medical recommendations for tracking lung nodules. For a stable 5mm nodule, they recommend **one more follow-up CT in 12 months** to confirm long-term stability.
- After that 12-month follow-up (if still stable), you'll likely be **done with surveillance** for this nodule.
- The **mild atelectasis** is nothing to worry about — it happens to almost everyone lying flat in a CT scanner.

---

## ❓ Questions to Ask Your Doctor

1. **"Since it's stable, does that mean it's not cancer?"** — Stability over 6+ months is very reassuring, and your doctor can discuss the very low likelihood of malignancy.
2. **"Do I need a follow-up CT in 12 months?"** — The report recommends this; confirm the timeline with your doctor.
3. **"What could this nodule be?"** — Small stable nodules are often old scars, tiny lymph nodes, or benign growths.
4. **"Should I be worried about anything in the meantime?"** — Ask about symptoms that should prompt earlier follow-up.

---

> ⚕️ *Remember: This explanation is for your understanding only. Always discuss your results with your treating physician who has your full medical history.*
"""
}


# ─── Model Backend ────────────────────────────────────────────────────────────

class MedGemmaBackend:
    """Handles model loading and inference for MedGemma."""

    def __init__(self, mode="demo"):
        self.mode = mode
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load the model based on mode."""
        if self.mode == "demo":
            print("🎭 Running in DEMO mode (pre-computed explanations)")
            return True

        if self.mode == "full":
            return self._load_transformers()

        return False

    def _load_transformers(self):
        """Load MedGemma via HuggingFace Transformers."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            print(f"⏳ Loading MedGemma ({MODEL_PRESET}) via Transformers...")

            model_id = "google/medgemma-4b-it"

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
            print("✅ MedGemma loaded successfully!")
            return True

        except Exception as e:
            print(f"⚠️ Could not load full model: {e}")
            print("↩️ Falling back to demo mode.")
            self.mode = "demo"
            return True

    def generate(self, report_text: str) -> str:
        """Generate patient-friendly explanation."""
        if self.mode == "demo":
            # Return pre-computed explanation if we have one
            for key, sample in SAMPLE_REPORTS.items():
                if report_text.strip() == sample.strip():
                    return DEMO_EXPLANATIONS.get(key, "Demo explanation not available for this report.")
            return self._generate_generic_demo(report_text)

        # Full model inference
        return self._generate_full(report_text)

    def _generate_full(self, report_text: str) -> str:
        """Run actual MedGemma inference."""
        import torch

        prompt = REPORT_PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            report_text=report_text
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
            )

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response

    def _generate_generic_demo(self, report_text: str) -> str:
        """Generate a placeholder response for custom reports in demo mode."""
        return f"""## 🩺 Demo Mode — Custom Report Analysis

> ⚠️ **Note:** You are running in demo mode. For full AI-powered analysis, run with `--mode full` on a machine with GPU access, or use the Kaggle notebook.

**Your report has been received** ({len(report_text)} characters).

In **full mode**, MyHealth Decoder would:
1. 📖 Break down every medical term into simple language
2. 🔍 Highlight what's normal vs. what needs attention
3. 💡 Explain the practical implications for your daily life
4. ❓ Generate 3-5 specific questions to ask your doctor

---

To get a real analysis:
- **Kaggle:** Run the competition notebook with GPU enabled
- **Local:** `python app.py --mode full` (requires GPU + ~8GB VRAM)

> ⚕️ *Always discuss your medical results with your treating physician.*
"""


# ─── Gradio UI ────────────────────────────────────────────────────────────────

def build_ui(backend: MedGemmaBackend):
    """Build and return the Gradio interface."""
    try:
        import gradio as gr
    except ImportError:
        print("❌ Gradio not installed. Install with: pip install gradio")
        return None

    mode_badge = "🟢 Full Model" if backend.mode == "full" else "🟡 Demo Mode"

    def analyze_report(report_text, report_type):
        """Main analysis function."""
        if not report_text or not report_text.strip():
            return "⚠️ Please paste a medical report or select a sample."

        return backend.generate(report_text)

    def load_sample(sample_name):
        """Load a sample report into the text area."""
        return SAMPLE_REPORTS.get(sample_name, "")

    with gr.Blocks(
        title="MyHealth Decoder",
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
        css="""
        .header { text-align: center; margin-bottom: 1rem; }
        .disclaimer {
            background: #FFF3CD; border: 1px solid #FFE69C;
            border-radius: 8px; padding: 12px; margin-top: 1rem;
            color: #664D03; font-size: 0.9em;
        }
        .badge {
            display: inline-block; padding: 4px 12px;
            border-radius: 12px; font-size: 0.85em; font-weight: 600;
        }
        """
    ) as demo:

        gr.HTML(f"""
        <div class="header">
            <h1>🏥 MyHealth Decoder</h1>
            <p style="font-size: 1.1em; color: #666;">
                Your AI medical report translator — powered by
                <strong>MedGemma</strong> (HAI-DEF)
            </p>
            <p><span class="badge" style="background: {'#D1FAE5' if backend.mode == 'full' else '#FEF3C7'};">
                {mode_badge}
            </span></p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Input")

                sample_dropdown = gr.Dropdown(
                    choices=list(SAMPLE_REPORTS.keys()),
                    label="Try a sample report",
                    value=None,
                    interactive=True,
                )

                report_input = gr.Textbox(
                    label="Paste your medical report here",
                    placeholder="Paste the text of any medical report...",
                    lines=15,
                    max_lines=30,
                )

                analyze_btn = gr.Button(
                    "🔍 Decode My Report",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                gr.Markdown("### 📖 Your Explanation")
                output = gr.Markdown(
                    value="*Select a sample report or paste your own, then click 'Decode My Report'.*"
                )

        gr.HTML("""
        <div class="disclaimer">
            <strong>⚕️ Important Disclaimer:</strong> MyHealth Decoder is an educational tool
            designed to help you understand medical terminology. It does <strong>NOT</strong>
            provide medical advice, diagnoses, or treatment recommendations. Always consult
            your healthcare provider for medical decisions.
        </div>
        """)

        # Wire up events
        sample_dropdown.change(
            fn=load_sample,
            inputs=[sample_dropdown],
            outputs=[report_input],
        )

        analyze_btn.click(
            fn=analyze_report,
            inputs=[report_input, sample_dropdown],
            outputs=[output],
        )

    return demo


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MyHealth Decoder — Patient Advocate")
    parser.add_argument(
        "--mode", choices=["demo", "full"], default="demo",
        help="'demo' uses pre-computed examples (no GPU needed). 'full' loads MedGemma."
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port for the Gradio server."
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio link (useful for demos)."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🏥  MyHealth Decoder — Patient Advocate")
    print(f"    Mode: {args.mode.upper()}")
    print("=" * 60)

    # Initialize backend
    backend = MedGemmaBackend(mode=args.mode)
    if not backend.load():
        print("❌ Failed to initialize backend. Exiting.")
        return

    # Build and launch UI
    demo = build_ui(backend)
    if demo is None:
        print("❌ Could not build UI. Make sure gradio is installed.")
        return

    print(f"\n🚀 Launching on http://localhost:{args.port}")
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
