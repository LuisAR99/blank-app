import streamlit as st

st.title("HW Manager")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

st.set_page_config(page_title="Multi-Page Labs", page_icon="🧪")

HW1 = st.Page("HW1.py", title="HW1")
HW2 = st.Page("HW2.py", title="HW2")

nav = st.navigation(pages=[HW2, HW1])   
nav.run()
