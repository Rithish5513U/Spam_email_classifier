import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

tfid = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')
ps = PorterStemmer()

def transformer(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    l = []
    for i in text:
        if i.isalnum():
            l.append(i)
    
    text = l[:]
    l = []
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            l.append(i)
    text = l[:]
    l = []

    for i in text:
        l.append(ps.stem(i))
    
    text = l[:]
    l = []

    return " ".join(text)

st.title("Spam Email Classifier")
email = st.text_area("Enter message ")

if st.button('Predict'):
    text = transformer(email)
    vector = tfid.transform([text])
    result = model.predict(vector)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Normal")

