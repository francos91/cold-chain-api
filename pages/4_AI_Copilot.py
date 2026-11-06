import streamlit as st
import requests

st.set_page_config(page_title="AI Operations Copilot", layout="wide", page_icon="🤖")

st.title("🤖 Autonomous Cold Chain Operations Copilot")
st.markdown(
    "Powered by **Llama 3.3 70B on Groq LPUs** with native tool-calling into our "
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
                # Groq responds in <1.5s; 25s is safe for Render cold starts
                res = requests.post(API_AGENT_URL, json=payload, timeout=25)
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
    st.rerun()  with this:import streamlit as st
import requests

st.set_page_config(page_title="Autonomous Copilot", layout="wide")
st.title("🤖 Autonomous Cold Chain Operations Copilot")

BACKEND_URL = "https://cold-chain-api-docker.onrender.com"

# 1. ADD ROLE SELECTOR TO SIDEBAR
st.sidebar.header("Copilot Settings")
user_role = st.sidebar.radio(
    "Your Role (Adapts AI Tone & Detail):", 
    ["Operations Manager", "Executive", "Dispatcher"],
    help="Executive gives high-level ZAR summaries. Dispatcher gives granular truck/route details."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 2. ADD LLMOps FEEDBACK TO ASSISTANT MESSAGES
        if msg["role"] == "assistant":
            # Provide a unique key for each feedback widget based on message index
            feedback = st.feedback("thumbs", key=f"feedback_{i}")
            if feedback is not None:
                # In a full production app, you would POST this to Supabase here to track model quality
                st.toast("Feedback recorded for model training!")

if prompt := st.chat_input("Ask about active routes, breach risks, weather, or fleet spend..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Format history for backend
    history_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m.get("content")
    ]

    with st.chat_message("assistant"):
        with st.spinner("Analyzing fleet telemetry..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/api/v1/agent/query",
                    json={
                        "user_query": prompt,
                        "chat_history": history_payload,
                        "user_role": user_role  # 3. PASS THE ROLE TO THE BACKEND
                    },
                    timeout=35
                )
                if response.status_code == 200:
                    bot_reply = response.json().get("response", "")
                    st.markdown(bot_reply)
                    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                    st.rerun() # Force rerun to render the feedback widget for the new message
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to Operations Copilot: {e}")
