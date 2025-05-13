import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re
import emoji
import contractions
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load BiLSTM model and preprocessing tools
model = tf.keras.models.load_model("Artifacts/BiLSTM.h5")

with open("Artifacts/bi_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = emoji.demojize(text)
    text = text.replace(":", "").replace("_", " ")
    text = contractions.fix(text)
    words = text.split()
    processed_words = []
    for word in words:
        clean = word.strip(string.punctuation)
        processed_words.append(clean)
    text = " ".join(processed_words)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Initialize the LLM (Gemini 2.0 Flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=st.secrets["GOOGLE_API_KEY"]
    temperature=0.2,
    max_output_tokens=2000
)

# Streamlit UI
st.title("Egypt Tourism Sentiment Classifier")
review = st.text_area("Enter a review about your visit to Egypt:")

if st.button("Predict Sentiment"):
    if review:
        clean_review = preprocess_text(review)
        seq = tokenizer.texts_to_sequences([clean_review])
        padded = pad_sequences(seq, maxlen=384, padding="post", truncating="post")
        prediction = model.predict(padded)
        sentiment = 'positive' if prediction[0][0] >= 0.5 else 'negative'

        if sentiment == 'positive':
            st.success(f"**Predicted Sentiment:** {sentiment.capitalize()}")
        else:
            st.error(f"**Predicted Sentiment:** {sentiment.capitalize()}")

        if sentiment == 'negative':
            with st.spinner('Generating recommendations...'):
                prompt = f"You are an expert in tourism management. Based on the following negative review about a tourist site in Egypt: '{review}', identify the specific issues raised by the reviewer. Then, provide 2-4 concise, practical, and actionable recommendations for Egyptian tourism authorities or site managers to address these issues and enhance the visitor experience. Ensure the solutions are realistic and tailored to the problems mentioned."
                message = HumanMessage(content=prompt)
                try:
                    response = llm.invoke([message])
                    st.success("**Recommendations:**")
                    st.write(response.content)
                except Exception as e:
                    st.error(f"An error occurred while generating recommendations: {e}")
        else:
            st.info("No recommendations needed for positive reviews.")
