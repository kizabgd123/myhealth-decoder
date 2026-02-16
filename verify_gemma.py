import os
import keras_nlp
import keras

# Set backend to JAX or TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"

print("✅ Keras backend set to TensorFlow")

try:
    # Attempt to load the Gemma 2B model
    # Note: Requires Kaggle authentication via ~/.kaggle/kaggle.json (which exists)
    print("⏳ Loading Gemma 2B model (gemma_2_2b_en)... this may take a while...")
    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2_2b_en")
    print("✅ Model loaded successfully!")

    # Test inference
    prompt = "Explain standard deviation to a 5 year old."
    print(f"📝 Prompt: {prompt}")
    output = gemma_lm.generate(prompt, max_length=64)
    print(f"🤖 Output: {output}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("💡 ensure you have accepted the license at https://www.kaggle.com/models/google/gemma")
