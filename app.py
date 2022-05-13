import streamlit as st
from multiapp import MultiApp
from app_pages import page1, page2 # import your app modules here

app = MultiApp()

st.markdown("""
# Calculer le scoring bancaire d'un prospect
Cette application utilise le [streamlit-multiapps](https://github.com/upraneelnihar/streamlit-multiapps) framework développé par [Praneel Nihar](https://medium.com/@u.praneel.nihar).
""")

# Add all your application here
app.add_app("Aperçu de la population", page1.app)
app.add_app("Fiche Prospect", page2.app)

# The main app
app.run()