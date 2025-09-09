import streamlit as st

st.title("HW Manager")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

st.set_page_config(page_title="Multi-Page Labs", page_icon="🧪")

lab2 = st.Page("HW1.py", title="HW1")
lab1 = st.Page("lab1.py", title="Lab 1")

nav = st.navigation(pages=[lab2, lab1])   
nav.run()
