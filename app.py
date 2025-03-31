import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.utils.validation import check_is_fitted

# Load model and vectorizer
try:
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    check_is_fitted(model)
except:
    st.error("🚨 Model not found! Run `train.py` first.")
    st.stop()

# Initialize PorterStemmer
ps = PorterStemmer()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# Streamlit UI
st.set_page_config(page_title="Email Spam Classifier", page_icon="📩")

st.markdown("<h1 style='text-align: center;'>📩 Email Spam Classifier</h1>", unsafe_allow_html=True)

input_sms = st.text_area("Enter the email text:", height=200)

if st.button("Check Spam"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message!")
    else:
        processed_sms = preprocess_text(input_sms)
        vector_input = vectorizer.transform([processed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.markdown("<h2 style='color: red; text-align: center;'>🚨 This is Spam!</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color: green; text-align: center;'>✅ Not Spam</h2>", unsafe_allow_html=True)
