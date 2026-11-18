from datetime import datetime, timezone
import requests
import streamlit as st

st.set_page_config(
    page_title="AI Operations Copilot", layout="wide", page_icon="🤖"
)

st.title("🤖 Autonomous Cold Chain Operations Copilot")
st.markdown(
    "Powered by **OpenAI gpt-oss-20b / 120b on Groq LPUs** with native"
    " tool-calling into our **Random Forest Inference Engine**, **Hybrid RRF"
    " Knowledge Base (pgvector)**, **Live Open-Meteo Weather**, and **Supabase"
    " Relational Telemetry**."
)

API_AGENT_URL = "https://cold-chain-api-docker.onrender.com/api/v1/agent/query"
BASE_AGENT_URL = API_AGENT_URL.rsplit("/query", 1)[0]

# 1. ADD ROLE SELECTOR TO SIDEBAR
st.sidebar.header("Copilot Settings")
user_role = st.sidebar.radio(
    "Your Role (Adapts AI Tone & Detail):",
    ["Operations Manager", "Executive", "Dispatcher"],
    help=(
        "Executive gives high-level ZAR summaries. Dispatcher gives granular"
        " truck/route details."
    ),
)

# Quick action sidebar
st.sidebar.header("🎯 Autonomous Tool Triggers")
example_queries = [
    "Summary of fleet",
    "Are there any active shipments exceeding 8.0°C right now?",
    (
        "What are our tax-free emission allowances under the South African"
        " carbon tax legislation for heavy transport fleets?"
    ),
    (
        "How do our transport strategy trade-offs look when comparing pure"
        " in-house distribution against third-party logistics (3PL) models?"
    ),
    (
        "What do our retail vehicle routing studies say about integrating"
        " backhauls to reduce empty mileage, and what is our current active"
        " fleet spend?"
    ),
    (
        "If I dispatch a truck from Johannesburg to Durban delayed by 8 hours,"
        " what is its breach risk?"
    ),
    "What is the current ambient temperature in Beaufort West?",
    (
        "If we shift 35% of freight on Natcor from road to rail, how much"
        " carbon tax do we save?"
    ),
]

# Maintain chat history
if "chat_messages" not in st.session_state:
  st.session_state.chat_messages = [{
      "role": "assistant",
      "content": (
          "👋 **Greetings.** I am your autonomous Cold Chain Copilot.\n\n"
          "I can:\n"
          "- Run live **Random Forest what-if breach risk predictions** on"
          " custom dispatches.\n"
          "- Search **South African Carbon Tax legislation, PPECB SOPs, and"
          " retail transport strategy research** via Hybrid RRF RAG.\n"
          "- Query **real-time ambient weather** along major freight"
          " corridors (Open-Meteo).\n"
          "- Audit **active temperature excursions (>8.0°C)** in the live"
          " fleet.\n"
          "- Calculate **modal-shift carbon tax savings** under South African"
          " rates (R28/tCO2).\n\n"
          "*How can I assist your fleet right now?*"
      ),
  }]


# --- HITL 60-SECOND VETO GUARDRAIL QUEUE ---
def render_guardrail_queue():
  try:
    r = requests.get(f"{BASE_AGENT_URL}/actions/pending", timeout=6)
    actions = r.json().get("actions", []) if r.status_code == 200 else []
  except Exception:
    actions = []

  if not actions:
    return

  st.warning(
      "🛡️ **ACTIVE AUTONOMOUS INTERVENTION GUARDRAIL (60s Human Veto Window)**"
  )
  now_utc = datetime.now(timezone.utc)
  for act in actions:
    action_id = act["id"]
    exp_str = act["expires_at"].replace("Z", "+00:00")
    remaining = max(
        0, int((datetime.fromisoformat(exp_str) - now_utc).total_seconds())
    )
    if remaining <= 0:
      continue

    with st.container(border=True):
      c1, c2 = st.columns([3, 1])
      with c1:
        st.markdown(
            f"**Shipment `{act['shipment_id']}`** ({act['corridor']}) — Risk:"
            f" **{act.get('risk_score', 'N/A')}%**\n\n📋 **Proposed Action:**"
            f" {act['proposed_action']}"
        )
        st.progress(
            min(1.0, remaining / 60.0),
            text=f"Auto-executes in {remaining}s unless vetoed...",
        )
      with c2:
        if st.button(
            "⛔ VETO ACTION",
            key=f"veto_{action_id}",
            type="primary",
            use_container_width=True,
        ):
          v_res = requests.post(
              f"{BASE_AGENT_URL}/actions/{action_id}/veto", timeout=8
          )
          if v_res.status_code == 200:
            st.toast(f"⛔ Action for {act['shipment_id']} VETOED!")
            st.rerun()


render_guardrail_queue()

# Render existing chat history
for i, msg in enumerate(st.session_state.chat_messages):
  with st.chat_message(msg["role"]):
    st.markdown(msg["content"])
    # 2. ADD LLMOps FEEDBACK TO ASSISTANT MESSAGES (Skip the welcome message at index 0)
    if msg["role"] == "assistant" and i > 0:
      feedback = st.feedback("thumbs", key=f"feedback_{i}")
      if feedback is not None:
        st.toast("Feedback recorded for model training!")


def submit_query(prompt_text: str):
  st.session_state.chat_messages.append(
      {"role": "user", "content": prompt_text}
  )
  with st.chat_message("user"):
    st.markdown(prompt_text)

  # 3. FORMAT HISTORY FOR BACKEND MEMORY
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
            "user_role": user_role,  # 4. PASS THE ROLE TO THE BACKEND
        }
        # 60s timeout accommodates Render cold starts, hybrid pgvector RAG, and 120b failover
        res = requests.post(API_AGENT_URL, json=payload, timeout=60)
        if res.status_code == 200:
          ans = res.json().get("response", "No response received.")
        else:
          ans = f"⚠️ API Error ({res.status_code}): {res.text}"
      except Exception as e:
        ans = f"⚠️ Failed to communicate with API: {e}"

      st.markdown(ans)
      st.session_state.chat_messages.append(
          {"role": "assistant", "content": ans}
      )


# Execute Quick Actions
for q in example_queries:
  if st.sidebar.button(q):
    submit_query(q)
    st.rerun()

# User Input Box
if user_prompt := st.chat_input(
    "Ask an operational question or what-if scenario..."
):
  submit_query(user_prompt)
  st.rerun()
