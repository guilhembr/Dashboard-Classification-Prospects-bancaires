import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def app():
    st.title("Page 3")

    st.write("This is the `Page 3` of this multi-page app.")

    st.write("In this app, we will be building a simple classification model using the Iris dataset.")