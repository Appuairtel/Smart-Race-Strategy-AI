
import streamlit as st
import pandas as pd

st.title("🏎 Smart Race Strategy AI Assistant")

uploaded_file = st.file_uploader("Upload race data CSV")

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    st.subheader("Race Data")
    st.write(data)

    tyre_age = data["TyreAge"].iloc[-1]
    weather = data["Weather"].iloc[-1]

    st.subheader("AI Recommendation")

    if tyre_age > 4:
        st.success("Pit stop recommended")

    elif weather == "Rain":
        st.warning("Switch to wet tyres")

    else:
        st.info("Continue current strategy")
