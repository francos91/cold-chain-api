import streamlit as st
import requests

st.set_page_config(page_title="AI Operations Copilot", layout="wide", page_icon="🤖")

st.title("🤖 Autonomous Cold Chain Operations Copilot")
st.markdown(
    "Powered by **Llama 3.3 70B on Groq LPUs** with native tool-calling into our "
    "**Random Forest Inference Engine**, **Live Open-Meteo Weather**, and **Supabase Relational Telemetry**."
)

API_AGENT_URL = "https://cold-chain-api-docker.onrender.com/api/v1/agent/query"

# 1. ADD ROLE SELECTOR TO SIDEBAR
st.sidebar.header("Copilot Settings")
user_role = st.sidebar.radio(
    "Your Role (Adapts AI Tone & Detail):", 
    ["Operations Manager", "Executive", "Dispatcher"],
    help="Executive gives high-level ZAR summaries. Dispatcher gives granular truck/route details."
)

# Quick action sidebar
st.sidebar.header("🎯 Autonomous Tool Triggers")
example_queries = [
    "Summary of fleet",
    "Are there any active shipments exceeding 8.0°C right now?",
    "If I dispatch a truck from Johannesburg to Durban delayed by 8 hours, what is its breach risk?",
    "What is the current ambient temperature in Beaufort West?",
    "If we shift 35% of freight on Natcor from road to rail, how much carbon tax do we save?",
]

# Maintain unified chat history state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Greetings.** I am your autonomous Cold Chain Copilot.\n\n"
                "I can:\n"
                "- Run live **Random Forest what-if breach risk predictions** on custom dispatches.\n"
                "- Query **real-time ambient weather** along major freight corridors (Open-Meteo).\n"
                "- Audit **active temperature excursions (>8.0°C)** in the live fleet.\n"
                "- Calculate **modal-shift carbon tax savings** under South African rates (R28/tCO2).\n\n"
                "*How can I assist your fleet right now?*"
            ),
        }
    ]

# Render existing chat history & LLMOps feedback widgets
for i, msg in enumerate(st.session_state.chat_messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i > 0:
            feedback = st.feedback("thumbs", key=f"feedback_{i}")
            if feedback is not None:
                st.toast("Feedback recorded for model training!")

def submit_query(prompt_text: str):
    st.session_state.chat_messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # Format recent history for backend memory payload
    history_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_messages[:-1]
        if m.get("content")
    ]

    with st.chat_message("assistant"):
        with st.spinner("🤖 Autonomous Agent analyzing fleet telemetry..."):
            try:
                payload = {
                    "user_query": prompt_text,
                    "chat_history": history_payload,
                    "user_role": user_role
                }
                res = requests.post(API_AGENT_URL, json=payload, timeout=35)
                if res.status_code == 200:
                    ans = res.json().get("response", "No response received.")
                else:
                    ans = f"⚠️ API Error ({res.status_code}): {res.text}"
            except Exception as e:
                ans = f"⚠️ Failed to communicate with API: {e}"

            st.markdown(ans)
            st.session_state.chat_messages.append({"role": "assistant", "content": ans})

# Execute Quick Actions from Sidebar
for q in example_queries:
    if st.sidebar.button(q):
        submit_query(q)
        st.rerun()

# User Chat Input Box
if user_prompt := st.chat_input("Ask an operational question or what-if scenario..."):
    submit_query(user_prompt)
    st.rerun()
