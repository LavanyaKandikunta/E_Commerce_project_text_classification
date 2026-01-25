#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import gdown, zipfile

app = Flask(__name__)

# ============================================================
# 0️⃣ Environment setup for Render Free/Starter Tier
# ============================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ============================================================
# 1️⃣ GRU model setup
# ============================================================
MODEL_DIR = "DL_models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE_ID = "1JsCHylh1wStYpPUQCHfMFRrcPYdBM11T"
METADATA_FILE_ID = "1a_txG7ddnCViQ8bAuI_UPkrBRTqpsOEq"

MODEL_FILE = os.path.join(MODEL_DIR, "gru_model.keras")
METADATA_FILE = os.path.join(MODEL_DIR, "metadata.json")

if not os.path.exists(MODEL_FILE):
    print("⚡ Downloading GRU model (.keras)...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_FILE, quiet=False)

if not os.path.exists(METADATA_FILE):
    print("⚡ Downloading metadata.json...")
    gdown.download(f"https://drive.google.com/uc?id={METADATA_FILE_ID}", METADATA_FILE, quiet=False)

gru_model = None
tokenizer = None
label_encoder = None

def load_gru_model():
    global gru_model, tokenizer, label_encoder
    if gru_model is None:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.text import tokenizer_from_json

        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        tokenizer = tokenizer_from_json(json.dumps(metadata["tokenizer"]))
        label_encoder = metadata["label_encoder"]

        gru_model = load_model(MODEL_FILE)
        print("✅ GRU model loaded successfully.")
    return gru_model

def predict_text_gru(text):
    load_gru_model()
    seq = tokenizer.texts_to_sequences([text])
    seq = np.array(seq)
    pred_probs = gru_model.predict(seq)
    pred_index = np.argmax(pred_probs, axis=1)[0]
    return label_encoder.get(str(pred_index), "Unknown")


# ============================================================
# 2️⃣ BERT model setup (with .safetensors and label_encoder.pkl)
# ============================================================
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import joblib

BERT_MODEL_DIR = "DL_models/bert_finetuned"
LABEL_ENCODER_FILE = os.path.join(BERT_MODEL_DIR, "label_encoder.pkl")

bert_tokenizer = None
bert_model = None
bert_label_map = None

def load_bert_model():
    global bert_model, bert_tokenizer, bert_label_map
    if bert_model is None:
        print("⚡ Loading fine-tuned BERT model (.safetensors)...")
        bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR)
        bert_model = BertForSequenceClassification.from_pretrained(
            BERT_MODEL_DIR,
            from_tf=False,
            torch_dtype=torch.float32,
        )
        bert_model.eval()

        # Load label encoder
        if os.path.exists(LABEL_ENCODER_FILE):
            bert_label_map = joblib.load(LABEL_ENCODER_FILE)
        else:
            bert_label_map = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}

        print("✅ Fine-tuned BERT model loaded successfully.")
    return bert_model

def predict_text_bert(text):
    load_bert_model()
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=1).item()
    return bert_label_map.get(pred_id, "Unknown")


# ============================================================
# 3️⃣ Recommendation system
# ============================================================
user_product_matrix = None
if os.path.exists("user_product_matrix.csv"):
    user_product_matrix = pd.read_csv("user_product_matrix.csv", index_col=0)
elif os.path.exists("user_product_matrix.zip"):
    with zipfile.ZipFile("user_product_matrix.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    user_product_matrix = pd.read_csv("user_product_matrix.csv", index_col=0)

def recommend_products(user_name, top_n=5):
    if user_product_matrix is None or user_product_matrix.empty:
        return ["⚠️ user_product_matrix not loaded"]
    if user_name not in user_product_matrix.index:
        return [f"⚠️ User '{user_name}' not found"]

    user_vec = user_product_matrix.loc[user_name].values.reshape(1, -1)
    sim_scores = cosine_similarity(user_vec, user_product_matrix)[0]
    sim_df = pd.DataFrame({"user": user_product_matrix.index, "similarity": sim_scores}).sort_values(by="similarity", ascending=False)
    similar_users = sim_df[sim_df["user"] != user_name].head(10)["user"]
    similar_user_data = user_product_matrix.loc[similar_users]
    recommended_scores = similar_user_data.sum(axis=0)
    user_products = user_product_matrix.loc[user_name]
    not_bought = recommended_scores[user_products == 0]
    return not_bought.sort_values(ascending=False).head(top_n).index.tolist()


# ============================================================
# 4️⃣ Flask Routes
# ============================================================
@app.route("/", methods=["GET", "POST"])
def home():
    prediction_gru = None
    prediction_bert = None
    recommendations = []
    if request.method == "POST":
        text_input = request.form.get("text_input")
        user_name = request.form.get("user_name")

        if text_input:
            prediction_gru = predict_text_gru(text_input)
            prediction_bert = predict_text_bert(text_input)

        if user_name:
            recommendations = recommend_products(user_name)
    return render_template("index.html",
                           prediction_gru=prediction_gru,
                           prediction_bert=prediction_bert,
                           recommendations=recommendations)

@app.route("/predict/gru", methods=["POST"])
def api_gru():
    data = request.get_json()
    text = data.get("text", "")
    return jsonify({"model": "GRU", "prediction": predict_text_gru(text)})

@app.route("/predict/bert", methods=["POST"])
def api_bert():
    data = request.get_json()
    text = data.get("text", "")
    return jsonify({"model": "BERT", "prediction": predict_text_bert(text)})

@app.route("/recommend", methods=["POST"])
def api_recommend():
    data = request.get_json()
    user = data.get("user")
    recs = recommend_products(user)
    return jsonify({"user": user, "recommendations": recs})

@app.route("/status")
def status():
    return jsonify({
        "gru_loaded": gru_model is not None,
        "bert_loaded": bert_model is not None,
        "users": len(user_product_matrix) if user_product_matrix is not None else 0
    })


# ============================================================
# 5️⃣ Run app with Waitress (production-ready)
# ============================================================
if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting app on 0.0.0.0:{port}", flush=True)
    serve(app, host="0.0.0.0", port=port)

