import streamlit as st
import requests

st.set_page_config(page_title="AI Operations Copilot", layout="wide", page_icon="🤖")

st.title("🤖 Autonomous Cold Chain Operations Copilot")
st.markdown(
    "Powered by **Gemini 2.5 Flash** with native tool-calling into our "
    "**Random Forest Inference Engine**, **Live Open-Meteo Weather**, and **Supabase Relational Telemetry**."
)

API_AGENT_URL = "https://cold-chain-api-docker.onrender.com/api/v1/agent/query"

# Maintain chat history
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Greetings, Operations Manager.** I am your autonomous Cold Chain Copilot.\n\n"
                "I can:\n"
                "- Run live **Random Forest what-if breach risk predictions** on custom dispatches.\n"
                "- Query **real-time ambient weather** along major freight corridors (Open-Meteo).\n"
                "- Audit **active temperature excursions (>8.0°C)** in the live fleet.\n"
                "- Calculate **modal-shift carbon tax savings** under South African rates (R28/tCO2).\n\n"
                "*How can I assist your fleet right now?*"
            ),
        }
    ]

# Render existing chat history
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Quick action sidebar
st.sidebar.header("🎯 Autonomous Tool Triggers")
example_queries = [
    "Are there any active shipments exceeding 8.0°C right now?",
    "If I dispatch a truck from Johannesburg to Durban delayed by 8 hours, what is its breach risk?",
    "What is the current ambient temperature in Beaufort West?",
    "If we shift 35% of freight on Natcor from road to rail, how much carbon tax do we save?",
]

def submit_query(prompt_text: str):
    st.session_state.chat_messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Autonomous Agent invoking internal tools..."):
            try:
                payload = {"user_query": prompt_text}
                res = requests.post(API_AGENT_URL, json=payload, timeout=30)
                if res.status_code == 200:
                    ans = res.json().get("response", "No response received.")
                else:
                    ans = f"⚠️ API Error ({res.status_code}): {res.text}"
            except Exception as e:
                ans = f"⚠️ Failed to communicate with API: {e}"

            st.markdown(ans)
            st.session_state.chat_messages.append({"role": "assistant", "content": ans})

for q in example_queries:
    if st.sidebar.button(q):
        submit_query(q)
        st.rerun()

# User Input Box
if user_prompt := st.chat_input("Ask an operational question or what-if scenario..."):
    submit_query(user_prompt)
    st.rerun()
