
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
df = pd.read_csv("chatbot_cleaned_with_embeddings.csv")
df["embedding"] = df["embedding_list"].apply(eval).apply(torch.tensor)
embedding_matrix = torch.stack(df["embedding"].tolist())
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing
stopwords = malaya.text.function.get_stopwords()
correction = malaya.spelling_correction.probability.Probability(corpus=stopwords)

def clean_text(text):
    text = str(text).lower()
    try:
        text = correction.correct(text)
    except:
        pass
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in stopwords]
    return " ".join(tokens)

# Load T5 model with fallback
try:
    tokenizer = T5Tokenizer.from_pretrained("malay-huggingface/t5-small-bahasa-cased")
    model = T5EncoderModel.from_pretrained("malay-huggingface/t5-small-bahasa-cased")
    model = model.to("cpu").eval()

    def get_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            return model(**inputs).last_hidden_state.mean(dim=1).squeeze()
except:
    def get_embedding(text):
        return torch.zeros(768)

# Chatbot logic
def chatbot_match(query):
    query_clean = clean_text(query)
    tfidf_vec = vectorizer.transform([query_clean])
    tfidf_scores = cosine_similarity(tfidf_vec, vectorizer.transform(df["clean_question"].fillna("")))[0]
    embed = get_embedding(query_clean)
    deep_scores = cosine_similarity(embed.unsqueeze(0), embedding_matrix)[0]
    combined_scores = 0.7 * deep_scores + 0.3 * tfidf_scores
    top_index = combined_scores.argmax()
    if combined_scores[top_index] < 0.3:
        return "Maaf, saya tidak pasti jawapannya. Sila cuba tanya dengan ayat lain."
    return df.iloc[top_index]["answer"]

# UI
with gr.Blocks(css=".gradio-container {max-width: 700px; margin: auto;}") as demo:
    gr.Image("logo-chatbot.png", width=80)
    gr.Markdown("## MARbot - UiTM Chatbot\nTanya apa-apa berkaitan pendaftaran pelajar baharu UiTM di sini!")
    inp = gr.Textbox(label="Tanya soalan anda:")
    out = gr.Textbox(label="Jawapan:")
    btn = gr.Button("Hantar")
    btn.click(fn=chatbot_match, inputs=inp, outputs=out)

demo.launch()
