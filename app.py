import streamlit as st
import pandas as pd
import plotly.express as px

from greenroute_engine import (
    train_demo_models,
    estimate_delivery_hours,
    carbon_footprint_kg,
    sla_risk_score,
    risk_label,
    recommend_routes
)

st.set_page_config(page_title="GreenRoute AI POC", layout="wide")

st.title("GreenRoute AI")
st.subheader("AI-Powered Green & Resilient Logistics Network - Blibli Case POC")

st.write(
    "This proof-of-concept demonstrates delivery delay prediction, SLA risk scoring, "
    "carbon footprint estimation, and carbon-aware route recommendation using synthetic data."
)

@st.cache_data
def load_data():
    return pd.read_csv("sample_shipments.csv")

df = load_data()
clf, reg, metrics = train_demo_models(df)

with st.sidebar:
    st.header("Shipment Scenario")
    vehicle_type = st.selectbox("Vehicle type", ["motorcycle", "van", "small_truck", "medium_truck"], index=1)
    distance_km = st.slider("Distance (km)", 5.0, 800.0, 120.0, 5.0)
    load_weight_kg = st.slider("Load weight (kg)", 1.0, 2000.0, 350.0, 10.0)
    traffic_index = st.slider("Traffic index", 0.0, 1.0, 0.55, 0.01)
    weather_risk = st.slider("Weather risk", 0.0, 1.0, 0.25, 0.01)
    hub_dwell_hours = st.slider("Hub dwell time (hours)", 0.0, 10.0, 2.5, 0.1)
    driver_score = st.slider("Driver behavior score", 0.0, 1.0, 0.82, 0.01)
    sla_hours = st.selectbox("SLA target (hours)", [6, 12, 24, 48, 72], index=2)

predicted_hours = estimate_delivery_hours(
    distance_km, vehicle_type, traffic_index, weather_risk, hub_dwell_hours, driver_score
)
carbon = carbon_footprint_kg(distance_km, vehicle_type, load_weight_kg)
risk = sla_risk_score(predicted_hours, sla_hours, traffic_index, weather_risk, hub_dwell_hours, driver_score)
risk_text = risk_label(risk)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted Delivery Time", f"{predicted_hours} hours")
c2.metric("SLA Risk Score", f"{risk}")
c3.metric("Risk Level", risk_text)
c4.metric("Carbon Estimate", f"{carbon} kgCO2")

if risk_text == "High":
    st.error("Recommended action: prioritize rerouting, reduce hub dwell time, or assign a faster vehicle.")
elif risk_text == "Medium":
    st.warning("Recommended action: monitor this shipment and prepare an alternative route.")
else:
    st.success("Recommended action: proceed with normal monitoring.")

st.divider()

st.header("Route Recommendation")
routes = recommend_routes(
    distance_km, vehicle_type, load_weight_kg, traffic_index, weather_risk,
    hub_dwell_hours, driver_score, sla_hours
)
st.dataframe(routes, use_container_width=True)

fig_route = px.bar(
    routes,
    x="route",
    y=["predicted_hours", "carbon_kgco2", "sla_risk_score"],
    barmode="group",
    title="Route Comparison: Time, Carbon, and SLA Risk"
)
st.plotly_chart(fig_route, use_container_width=True)

st.divider()

left, right = st.columns(2)

with left:
    st.header("Historical Shipment Overview")
    st.dataframe(df.head(20), use_container_width=True)

with right:
    st.header("Demo Model Metrics")
    st.write("The metrics below are calculated from the synthetic sample dataset.")
    st.json(metrics)

fig1 = px.scatter(
    df,
    x="distance_km",
    y="carbon_kgco2",
    color="vehicle_type",
    size="load_weight_kg",
    hover_data=["shipment_id", "risk_level"],
    title="Carbon Footprint by Distance and Vehicle Type"
)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.histogram(
    df,
    x="sla_risk_score",
    color="risk_level",
    title="Distribution of SLA Risk Scores"
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

st.header("Prototype Architecture")
st.markdown(
    """
    **Input:** GPS/IoT data, ERP/TMS shipment data, traffic API, weather API, hub dwell time, driver behavior, vehicle profile.

    **Process:** Data cleaning → delay prediction → carbon calculation → SLA risk scoring → route optimization.

    **Output:** Recommended route, predicted delay risk, carbon estimate, alert level, and operational action.
    """
)
