# 🏥 MyHealth Decoder

**Your AI Medical Report Translator** — Powered by [MedGemma](https://ai.google.dev/gemma/docs/medgemma) (HAI-DEF)

> Built for the [Med-Gemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge) on Kaggle.

## What It Does

Paste any medical report (radiology, lab work, pathology, CT scans) and get:
- ✅ **Plain-English explanation** of every finding
- 🔍 **Key findings breakdown** (normal vs. needs attention)
- 💡 **What it means for you** in practical terms
- ❓ **Questions to ask your doctor** — personalized to your report

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/myhealth-decoder.git
cd myhealth-decoder
pip install -r requirements.txt

# Run in demo mode (no GPU needed)
python app.py --mode demo

# Run with MedGemma (requires GPU + ~8GB VRAM)
python app.py --mode full
```

## Architecture

| Component | Technology |
|-----------|-----------|
| **Model** | MedGemma 4B Instruct (google/medgemma-4b-it) |
| **Backend** | HuggingFace Transformers |
| **UI** | Gradio |
| **Privacy** | 100% local inference — no data leaves your device |

## Sample Reports Included

- 🫁 Chest X-Ray (Radiology)
- 🩸 Blood Work (CBC + Metabolic Panel)
- 🔬 Pathology Report (Breast Biopsy)
- 📷 CT Scan (Pulmonary Nodule Follow-up)

## Privacy-First Design

All processing happens **on your device**. No patient data is sent to any external server.
This makes MyHealth Decoder suitable for:
- Hospital kiosks
- Rural clinics without internet
- Personal health literacy

## Edge AI Ready

The 4B model runs on consumer GPUs (~8GB VRAM in bfloat16).
With quantization (4-bit GPTQ), it fits in ~3GB — enabling deployment on tablets and smartphones.

## ⚕️ Disclaimer

MyHealth Decoder is an **educational tool** designed to improve health literacy.
It does **NOT** provide medical advice, diagnoses, or treatment recommendations.
Always consult your healthcare provider for medical decisions.

## License

Apache 2.0
