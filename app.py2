
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import requests

load_dotenv()

st.set_page_config(page_title="Race Strategy AI Copilot", layout="wide")

st.title("🏎️ Race Strategy AI Copilot")
st.write("AI assistant for racing strategy, pit stop advice, and fan explanation.")

uploaded_file = st.file_uploader("Upload race CSV data", type=["csv"])

def analyse_data(df):
    avg_laptime = df["laptime"].mean()
    last_laptime = df["laptime"].iloc[-1]
    tyre_age = df["tyre_age"].iloc[-1]
    tyre = df["tyre"].iloc[-1]
    position = df["position"].iloc[-1]
    weather = df["weather"].iloc[-1]

    if last_laptime > avg_laptime + 1.5 or tyre_age >= 5:
        risk = "High"
        advice = "Pit stop recommended soon."
    elif last_laptime > avg_laptime + 0.8:
        risk = "Medium"
        advice = "Monitor tyre degradation."
    else:
        risk = "Low"
        advice = "Stay out for now."

    return {
        "avg_laptime": round(avg_laptime, 2),
        "last_laptime": last_laptime,
        "tyre_age": tyre_age,
        "tyre": tyre,
        "position": position,
        "weather": weather,
        "risk": risk,
        "advice": advice
    }

def granite_prompt(summary):
    return f"""
You are an AI race strategy engineer.

Race data summary:
Average lap time: {summary['avg_laptime']}
Last lap time: {summary['last_laptime']}
Tyre: {summary['tyre']}
Tyre age: {summary['tyre_age']}
Weather: {summary['weather']}
Position: {summary['position']}
Risk level: {summary['risk']}
Rule-based advice: {summary['advice']}

Give:
1. Pit stop recommendation
2. Risk explanation
3. Simple fan-friendly explanation
4. Confidence level
"""

def call_granite(prompt):
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL")

    if not api_key or not project_id:
        return "Granite API details are missing. Add your IBM watsonx API key and project ID in the .env file."

    token_response = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={
            "apikey": api_key,
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
        }
    )

    access_token = token_response.json()["access_token"]

    endpoint = f"{url}/ml/v1/text/generation?version=2023-05-29"

    payload = {
        "model_id": "ibm/granite-13b-instruct-v2",
        "project_id": project_id,
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 400,
            "temperature": 0.3
        }
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(endpoint, json=payload, headers=headers)

    if response.status_code != 200:
        return f"IBM Granite API error: {response.text}"

    return response.json()["results"][0]["generated_text"]

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Race Data")
    st.dataframe(df)

    summary = analyse_data(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Level", summary["risk"])
    col2.metric("Tyre Age", summary["tyre_age"])
    col3.metric("Last Lap Time", summary["last_laptime"])

    st.subheader("Rule-Based Strategy Advice")
    st.success(summary["advice"])

    if st.button("Ask IBM Granite AI"):
        prompt = granite_prompt(summary)
        ai_response = call_granite(prompt)

        st.subheader("IBM Granite AI Recommendation")
        st.write(ai_response)

else:
    st.info("Upload sample_race_data.csv to start.")
