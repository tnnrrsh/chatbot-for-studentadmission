import gradio as gr
import torch
import pandas as pd
import joblib
import re
import malaya
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5EncoderModel

# Load assets
# Ensure these paths are correct relative to app.py in your Hugging Face Space
df = pd.read_csv("chatbot_cleaned_with_embeddings.csv")
df["embedding"] = df["embedding_list"].apply(eval).apply(torch.tensor)
embedding_matrix = torch.stack(df["embedding"].tolist())
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing
stopwords = malaya.text.function.get_stopwords()
# It's better to load the correction model outside of a try-except if possible,
# or handle the exception more gracefully if it's a critical component.
# For now, we will assume it loads correctly.
try:
    correction = malaya.spelling_correction.probability.Probability(corpus=stopwords)
except Exception as e:
    print(f"Error loading malaya spelling correction: {e}")
    # Define a dummy correction function if it fails to load
    def dummy_correct(text):
        print("Malaya spelling correction not loaded, skipping correction.")
        return text
    correction = type('DummyCorrection', (object,), {'correct': dummy_correct})()


def clean_text(text):
    text = str(text).lower()
    try:
        text = correction.correct(text)
    except Exception as e:
        # print(f"Error during spelling correction: {e}") # Log error but continue
        pass # Allow the cleaning to proceed even if correction fails
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in stopwords]
    return " ".join(tokens)

# Load T5 model with fallback - crucial for deployment
try:
    tokenizer = T5Tokenizer.from_pretrained("malay-huggingface/t5-small-bahasa-cased")
    model = T5EncoderModel.from_pretrained("malay-huggingface/t5-small-bahasa-cased")
    model = model.to("cpu").eval() # Ensure model is on CPU if not using GPU instance

    def get_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            return model(**inputs).last_hidden_state.mean(dim=1).squeeze()
except Exception as e:
    print(f"Error loading T5 model: {e}")
    print("Falling back to dummy embedding function. Chatbot performance may be affected.")
    # Fallback to zeros is problematic for cosine similarity.
    # A better fallback for deep_scores might be to set them to 0,
    # and rely only on TF-IDF, or return a consistent but non-zero vector.
    # For now, we'll keep the zeros as per your original, but be aware of its impact.
    def get_embedding(text):
        return torch.zeros(model.config.hidden_size if 'model' in locals() and model else 768)


# Chatbot logic
def chatbot_match(query):
    query_clean = clean_text(query)
    tfidf_vec = vectorizer.transform([query_clean])
    tfidf_scores = cosine_similarity(tfidf_vec, vectorizer.transform(df["clean_question"].fillna("")))[0]

    embed = get_embedding(query_clean)
    # Check if embed is all zeros (due to fallback) and handle deep_scores accordingly
    if torch.all(embed == 0):
        deep_scores = torch.zeros(embedding_matrix.shape[0]) # No deep scores if embedding failed
    else:
        deep_scores = cosine_similarity(embed.unsqueeze(0), embedding_matrix)[0]

    # Adjust combined_scores logic if deep_scores are unreliable
    if torch.all(embed == 0): # If T5 failed, rely more on TF-IDF or just TF-IDF
        combined_scores = tfidf_scores # Only TF-IDF if deep embedding failed
    else:
        combined_scores = 0.7 * deep_scores + 0.3 * tfidf_scores

    top_index = combined_scores.argmax()

    # Consider the threshold carefully if deep_scores might be 0
    # If using only TF-IDF, the threshold of 0.3 might need adjustment.
    # For now, keeping as is.
    if combined_scores[top_index] < 0.3:
        return "Maaf, saya tidak pasti jawapannya. Sila cuba tanya dengan ayat lain."
    return df.iloc[top_index]["answer"]

# UI
with gr.Blocks(css=".gradio-container {max-width: 700px; margin: auto;}") as demo:
    gr.Image("logo-chatbot.png", width=80) # Ensure logo-chatbot.png is in the same directory
    gr.Markdown("## MARbot - UiTM Chatbot\nTanya apa-apa berkaitan pendaftaran pelajar baharu UiTM di sini!")
    inp = gr.Textbox(label="Tanya soalan anda:")
    out = gr.Textbox(label="Jawapan:")
    btn = gr.Button("Hantar")
    btn.click(fn=chatbot_match, inputs=inp, outputs=out)
