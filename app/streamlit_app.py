"""Interactive percentile estimator.

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py

Uses only the exported JSON artifact, the same path a production backend takes.
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from jee_percentile.portable import load_artifact, predict_curve, predict_percentile

ARTIFACT = Path(__file__).resolve().parents[1] / "models" / "percentile_model_artifact.json"

st.set_page_config(page_title="JEE Main Percentile Predictor", page_icon="📈", layout="wide")
artifact = load_artifact(ARTIFACT)

st.title("📈 JEE Main Percentile Predictor")
st.caption(f"Cutoff-ensemble model · trained on {artifact['n_shifts_trained_on']} real shifts · "
           "leave-one-shift-out error ≈ 0.72 percentile points")

left, right = st.columns([1, 2], gap="large")
with left:
    marks = st.slider("Your raw marks (out of 300)", 0, 300, 140)
    st.markdown("**How hard was the Physics section?**")
    phy_hard = st.slider("Questions labelled hard (%)", 0, 20, 4)
    phy_easy = st.slider("Questions labelled easy (%)", 10, 70, 40)

percentiles, curve = predict_curve(artifact, phy_hard, phy_easy)
low, point, high = predict_percentile(artifact, marks, phy_hard, phy_easy)

with left:
    if marks >= curve[-1]:
        st.metric("Estimated percentile", "99+")
        st.caption("Above the 99th-percentile cutoff for this paper (the model covers 90–99).")
    elif marks <= curve[0]:
        st.metric("Estimated percentile", "< 90")
        st.caption("Below the 90th-percentile cutoff for this paper (the model covers 90–99).")
    else:
        st.metric("Estimated percentile", f"{point:.2f}")
        st.caption(f"80% band: **{low:.2f} – {high:.2f}** (back-tested coverage 80%)")

band = artifact.get("prediction_interval") or {"lower": [0] * 10, "upper": [0] * 10}
frame = pd.DataFrame({
    "percentile": percentiles,
    "marks": curve,
    "lo": [c + l for c, l in zip(curve, band["lower"])],
    "hi": [c + u for c, u in zip(curve, band["upper"])],
})
area = alt.Chart(frame).mark_area(opacity=0.18, color="#4F46E5").encode(
    x=alt.X("lo:Q", title="Raw marks"), x2="hi:Q", y=alt.Y("percentile:Q", scale=alt.Scale(domain=[89.5, 99.5])))
line = alt.Chart(frame).mark_line(point=True, color="#4F46E5", strokeWidth=3).encode(
    x="marks:Q", y="percentile:Q", tooltip=["percentile", alt.Tooltip("marks:Q", format=".1f")])
rule = alt.Chart(pd.DataFrame({"m": [marks]})).mark_rule(color="#E11D48", strokeDash=[4, 4]).encode(x="m:Q")

with right:
    st.subheader("Marks needed for each percentile on this paper")
    st.altair_chart((area + line + rule).properties(height=420), use_container_width=True)
    st.dataframe(frame[["percentile", "marks"]].rename(columns={"marks": "marks needed"}).round(1).set_index("percentile").T,
                 use_container_width=True)
