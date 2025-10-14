import streamlit as st

st.title("HW Manager")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

st.set_page_config(page_title="Multi-Page Labs", page_icon="🧪")

HW1 = st.Page("HW1.py", title="HW1")
HW2 = st.Page("HW2.py", title="HW2")
HW3 = st.Page("HW3.py", title="HW3")
HW4 = st.Page("HW4.py", title="HW4")
HW5 = st.Page("HW5.py", title="HW5")
HW6 = st.Page("HW7.py", title="HW7")

nav = st.navigation(pages=[HW5, HW4, HW3, HW2, HW1])   
nav.run()
