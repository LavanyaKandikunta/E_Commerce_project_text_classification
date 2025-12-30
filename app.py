#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[9]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/E_Commerce_project_text_classification')


# In[10]:


# Install dependencies
#!pip install flask pyngrok python-dotenv --quiet
import subprocess

subprocess.check_call(["pip", "install", "flask", "pyngrok", "python-dotenv"])


# In[11]:


from flask import Flask, request, jsonify
from pyngrok import ngrok
from dotenv import load_dotenv
from keras.models import load_model
import os
from threading import Thread
import time
import pandas as pd


# # Start the tunnel
# 
# 

# In[12]:


load_dotenv()
ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))


# In[13]:


# Kill previous ngrok tunnels
ngrok.kill()

# Kill any processes using port 5000
get_ipython().system('fuser -k 5000/tcp')


# In[16]:


# Create Flask app
app = Flask(__name__)

# This ensures Render finds the file inside the container, even make it universal:
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # for script
except NameError:
    BASE_DIR = os.getcwd()  # for notebook / interactive env

#Load user–product matrix.
USER_MATRIX_FILE = os.path.join(BASE_DIR, "user_product_matrix.csv")
user_product_matrix = pd.read_csv(USER_MATRIX_FILE, index_col=0)


# Load only the chosen model. This prevents the API from crashing if the file is missing.
# Build the path to your model file inside the repo
MODEL_PATH = os.path.join(BASE_DIR, "Models_results", "DL_models", "gru_model.keras")

# Load the model safely
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = None
    print("⚠️ Model file not found:", MODEL_PATH)


#model = load_model("/content/drive/MyDrive/E_Commerce_project_text_classification/Models_results/DL_models/gru_model.keras")


# In[17]:


from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load user–product matrix
user_product_matrix = pd.read_csv("user_product_matrix.csv", index_col=0)

def recommend_products(user_name, top_n=5):
    """
    Recommend products to a given user based on user-user collaborative filtering.

    """
    # Safety check
    if user_name not in user_product_matrix.index:
        print(f"⚠️ User '{user_name}' not found in dataset.")
        return []

    # Step 1: Compute similarity
    user_vector = user_product_matrix.loc[user_name].values.reshape(1, -1)
    similarity_scores = cosine_similarity(user_vector, user_product_matrix)[0]

    similarity_df = pd.DataFrame({
        'user': user_product_matrix.index,
        'similarity': similarity_scores
    }).sort_values(by='similarity', ascending=False)

    # Step 2: Select top similar users (excluding the same user)
    similar_users = similarity_df[similarity_df['user'] != user_name].head(10)['user']

    # Step 3: Aggregate products from top similar users
    similar_user_data = user_product_matrix.loc[similar_users]
    recommended_scores = similar_user_data.sum(axis=0)

    # Step 4: Exclude products already purchased by this user
    user_products = user_product_matrix.loc[user_name]
    not_bought = recommended_scores[user_products == 0]

    # Step 5: Return top N product names
    recommendations = not_bought.sort_values(ascending=False).head(top_n).index.tolist()
    return recommendations


# In[18]:


# Stop current Flask server
get_ipython().system('pkill -f flask')


# In[19]:


@app.route("/")
def home():
    return "E-Commerce Recommendation API is running 🚀"

# Define API route
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    user_name = data.get("user")
    top_n = data.get("top_n", 5)
    recs = recommend_products(user_name, top_n)
    return jsonify({"user": user_name, "recommendations": recs})

 #Start Flask in a background thread
def run_app():
    app.run(port=5000)

Thread(target=lambda: app.run(host='0.0.0.0', port=5000)).start()

# Wait for Flask to start
time.sleep(3)

# Start ngrok tunnel
public_url = ngrok.connect(5000).public_url #extracts just the public HTTPS URL as a string.
print("Ngrok public URL:", public_url)


# Flask is still running can't execute next code.
# 
# Reasons:
# 
# 1. app.run() blocks the Python process.
#   
#     That means our Python process stops at the Flask server and waits for requests.
# 
# 2. Nothing after app.run() executes until you stop the server manually (Ctrl+C in local Python, or interrupt in Colab).
# 
# 3. So our requests.post(...) code never executes because Flask is still running.
# 
# Solution: is to run Flask in a background using thread

# In[20]:


import requests, json

# Test the API using requests
url = f"{public_url}/recommend"
data = {"user": "Pink", "top_n": 5}

r = requests.post(url, json=data)
print("API response:")
print(json.dumps(r.json(), indent=2))


# Everything is now functioning end-to-end:
# 
# ✅ Flask backend
# 
# ✅ Ngrok tunnel
# 
# ✅ External API request
# 
# ✅ Recommendation response

# In[8]:


get_ipython().system('jupyter nbconvert --to python app.ipynb')


# In[ ]:




