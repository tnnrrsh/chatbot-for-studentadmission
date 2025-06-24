import os
import re
import torch
import malaya
import joblib
import pandas as pd
import gradio as gr
from transformers import T5Tokenizer, T5EncoderModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
df = pd.read_csv("chatbot_cleaned_with_embeddings.csv")
df["embedding"] = df["embedding_list"].apply(eval).apply(torch.tensor)
embedding_matrix = torch.stack(df["embedding"].tolist())

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

# Load TF-IDF Vectorizer
if os.path.exists("tfidf_vectorizer.pkl"):
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
else:
    vectorizer = TfidfVectorizer()
    vectorizer.fit(df["clean_question"].fillna(""))
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
tfidf_matrix = vectorizer.transform(df["clean_question"].fillna(""))

# Load T5 Model or fallback
try:
    tokenizer = T5Tokenizer.from_pretrained("malay-huggingface/t5-small-bahasa-cased")
    model = T5EncoderModel.from_pretrained("malay-huggingface/t5-small-bahasa-cased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    def get_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            return model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu()

except Exception as e:
    print("⚠️ Model fallback active. Using TF-IDF only.")
    def get_embedding(text):
        return torch.zeros(768)

# Chatbot Matcher
def chatbot_match(query, threshold=0.3, alpha=0.7, beta=0.3, top_k=1):
    query_clean = clean_text(query)
    tfidf_vec = vectorizer.transform([query_clean])
    tfidf_scores = cosine_similarity(tfidf_vec, tfidf_matrix)[0]
    embed = get_embedding(query_clean)
    deep_scores = cosine_similarity(embed.unsqueeze(0), embedding_matrix)[0]
    combined_scores = (alpha * deep_scores) + (beta * tfidf_scores)
    best_indices = combined_scores.argsort()[-top_k:][::-1]
    if combined_scores[best_indices[0]] < threshold:
        return "Maaf, saya tidak pasti jawapannya. Sila cuba tanya dengan ayat lain."
    return df.iloc[best_indices[0]]["answer"]

# Gradio Interface
chatbot = gr.ChatInterface(
    fn=chatbot_match,
    title="MARbot - UiTM Chatbot",
    description="Tanya apa-apa berkaitan pendaftaran pelajar baharu UiTM.",
    theme="soft",
    examples=[
        "Bagaimana saya nak daftar kursus?",
        "Boleh saya tahu bila orientasi bermula?",
        "Apa itu Ufuture?"
    ]
)

if __name__ == "__main__":
    chatbot.launch()
