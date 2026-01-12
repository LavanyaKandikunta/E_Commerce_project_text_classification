#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify, render_template
import os, json, numpy as np, pandas as pd, gdown
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import tflite_runtime.interpreter as tflite  # ✅ lightweight runtime

app = Flask(__name__)

# ================================
# 0️⃣ File paths and Google Drive IDs
# ================================
MODEL_DIR = "DL_models"
MODEL_FILE = os.path.join(MODEL_DIR, "gru_model.tflite")
METADATA_FILE = os.path.join(MODEL_DIR, "metadata.json")

MODEL_ID = "1ZF9JF-jPkBxAulDdIgkEksLf8dRnbchV"       # 🟢 update with your TFLite Drive ID
METADATA_ID = "1a_txG7ddnCViQ8bAuI_UPkrBRTqpsOEq"  # your metadata.json ID

os.makedirs(MODEL_DIR, exist_ok=True)

# ================================
# 1️⃣ Download model + metadata if missing
# ================================
if not os.path.exists(MODEL_FILE):
    print("⚡ Downloading GRU TFLite model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_FILE, quiet=False)

if not os.path.exists(METADATA_FILE):
    print("⚡ Downloading metadata.json from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={METADATA_ID}", METADATA_FILE, quiet=False)

print("✅ All model files ready.")

# ================================
# 2️⃣ Load metadata and TFLite model
# ================================
with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)
tokenizer = tokenizer_from_json(json.dumps(metadata["tokenizer"]))
label_encoder = metadata["label_encoder"]

interpreter = tflite.Interpreter(model_path=MODEL_FILE)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ================================
# 3️⃣ Predict function using TFLite
# ================================
def predict_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence, dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], sequence)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    pred_index = int(np.argmax(preds, axis=1)[0])
    return label_encoder.get(str(pred_index), "Unknown")

# ================================
# 4️⃣ Simple recommendation system
# ================================
user_product_matrix = None
if os.path.exists("user_product_matrix.csv"):
    user_product_matrix = pd.read_csv("user_product_matrix.csv", index_col=0)

def recommend_products(user_name, top_n=5):
    if user_product_matrix is None or user_product_matrix.empty:
        return ["⚠️ user_product_matrix not loaded"]
    if user_name not in user_product_matrix.index:
        return [f"⚠️ User '{user_name}' not found"]

    user_vector = user_product_matrix.loc[user_name].values.reshape(1, -1)
    similarity_scores = cosine_similarity(user_vector, user_product_matrix)[0]
    similarity_df = pd.DataFrame({
        "user": user_product_matrix.index,
        "similarity": similarity_scores
    }).sort_values(by="similarity", ascending=False)

    similar_users = similarity_df[similarity_df["user"] != user_name].head(10)["user"]
    similar_user_data = user_product_matrix.loc[similar_users]
    recommended_scores = similar_user_data.sum(axis=0)

    user_products = user_product_matrix.loc[user_name]
    not_bought = recommended_scores[user_products == 0]
    return not_bought.sort_values(ascending=False).head(top_n).index.tolist()

# ================================
# 5️⃣ Flask Routes
# ================================
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        text = request.form.get("text_input")
        if text:
            prediction = predict_text(text)
    return render_template("index.html", prediction=prediction)

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    label = predict_text(text)
    return jsonify({"text": text, "prediction": label})

@app.route("/recommend", methods=["POST"])
def recommend_api():
    data = request.get_json(force=True)
    user_name = data.get("user")
    top_n = data.get("top_n", 5)
    recs = recommend_products(user_name, top_n)
    return jsonify({"user": user_name, "recommendations": recs})

@app.route("/status")
def status():
    return jsonify({
        "tflite_model_loaded": True,
        "metadata_loaded": tokenizer is not None,
        "recommendation_data": user_product_matrix is not None
    })

# ================================
# 6️⃣ Run Flask
# ================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)


# In[ ]:




