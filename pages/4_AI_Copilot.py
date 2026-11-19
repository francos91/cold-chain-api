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
      c1, c2 = st.columns([3, 1.3])
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
        veto_reason = st.selectbox(
            "Veto Reason (RLHF):",
            options=[
                "Driver contacted / situational context",
                "Route cleared by weather/traffic update",
                "False positive risk score",
                "Depot slot rescheduled manually",
                "Other operational override",
            ],
            key=f"reason_{action_id}",
        )
        if st.button(
            "⛔ VETO ACTION",
            key=f"veto_{action_id}",
            type="primary",
            use_container_width=True,
        ):
          v_res = requests.post(
              f"{BASE_AGENT_URL}/actions/{action_id}/veto",
              json={"reason": veto_reason},
              timeout=8,
          )
          if v_res.status_code == 200:
            st.toast(
                f"⛔ Action for {act['shipment_id']} VETOED ({veto_reason})!"
            )
            st.rerun()
