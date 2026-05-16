import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

VEHICLE_SPEED = {
    "motorcycle": 28,
    "van": 38,
    "small_truck": 42,
    "medium_truck": 45,
}

FUEL_EFFICIENCY_KM_PER_L = {
    "motorcycle": 35,
    "van": 11,
    "small_truck": 7,
    "medium_truck": 5,
}

EMISSION_FACTOR_KG_CO2_PER_LITER = 2.31


def carbon_footprint_kg(distance_km, vehicle_type, load_weight_kg):
    """Estimate shipment-level carbon footprint.

    Formula:
    fuel_used_liter = distance_km / vehicle_fuel_efficiency
    load_factor = 1 + load_weight_kg / 4000
    carbon = fuel_used_liter * emission_factor * load_factor

    This is a simplified prototype formula. In production, it should be calibrated
    with actual fuel consumption, vehicle age, driving behavior, and route profile.
    """
    efficiency = FUEL_EFFICIENCY_KM_PER_L.get(vehicle_type, 10)
    fuel_liters = distance_km / efficiency
    load_factor = 1 + (load_weight_kg / 4000)
    return round(fuel_liters * EMISSION_FACTOR_KG_CO2_PER_LITER * load_factor, 2)


def estimate_delivery_hours(distance_km, vehicle_type, traffic_index, weather_risk, hub_dwell_hours, driver_score):
    """Estimate delivery time using interpretable logistics factors."""
    base_speed = VEHICLE_SPEED.get(vehicle_type, 38)
    adjusted_speed = base_speed * (1 - 0.55 * traffic_index) * (0.85 + 0.3 * driver_score)
    adjusted_speed = max(adjusted_speed, 8)
    travel_hours = distance_km / adjusted_speed
    return round(travel_hours + hub_dwell_hours + weather_risk * 3, 2)


def sla_risk_score(predicted_hours, sla_hours, traffic_index, weather_risk, hub_dwell_hours, driver_score):
    """Calculate SLA risk score from 0 to 1."""
    ratio = min(predicted_hours / max(sla_hours, 1), 2) / 2
    score = (
        0.25 * traffic_index
        + 0.20 * weather_risk
        + 0.25 * min(hub_dwell_hours / 8.5, 1)
        + 0.20 * ratio
        + 0.10 * (1 - driver_score)
    )
    return round(float(np.clip(score, 0, 1)), 3)


def risk_label(score):
    if score >= 0.65:
        return "High"
    if score >= 0.35:
        return "Medium"
    return "Low"


def train_demo_models(df):
    """Train small demo models from synthetic data."""
    features = [
        "distance_km", "load_weight_kg", "traffic_index", "weather_risk",
        "hub_dwell_hours", "driver_score", "sla_hours"
    ]
    encoded = pd.get_dummies(df[features + ["vehicle_type"]], columns=["vehicle_type"], drop_first=False)
    X = encoded
    y_delay = df["is_delayed"]
    y_carbon = df["carbon_kgco2"]

    X_train, X_test, y_train, y_test = train_test_split(X, y_delay, test_size=0.25, random_state=42)
    clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_carbon, test_size=0.25, random_state=42)
    reg = RandomForestRegressor(n_estimators=150, random_state=42)
    reg.fit(Xc_train, yc_train)
    pred_carbon = reg.predict(Xc_test)

    metrics = {
        "delay_accuracy": round(accuracy_score(y_test, pred), 3),
        "delay_f1_score": round(f1_score(y_test, pred), 3),
        "carbon_mae_kgco2": round(mean_absolute_error(yc_test, pred_carbon), 2),
        "feature_columns": list(X.columns)
    }
    return clf, reg, metrics


def recommend_routes(distance_km, vehicle_type, load_weight_kg, traffic_index, weather_risk, hub_dwell_hours, driver_score, sla_hours):
    """Generate three route scenarios and select the best one.

    The score balances delivery time, carbon emission, and SLA risk.
    Lower score is better.
    """
    scenarios = [
        {
            "route": "Fastest Route",
            "distance_multiplier": 1.10,
            "traffic_delta": -0.20,
            "description": "Slightly longer distance, lower congestion."
        },
        {
            "route": "Shortest Route",
            "distance_multiplier": 0.92,
            "traffic_delta": 0.18,
            "description": "Shortest distance, but higher congestion risk."
        },
        {
            "route": "Green Balanced Route",
            "distance_multiplier": 1.00,
            "traffic_delta": -0.05,
            "description": "Balanced distance, fuel use, and SLA risk."
        },
    ]

    results = []
    for s in scenarios:
        d = round(distance_km * s["distance_multiplier"], 1)
        t = float(np.clip(traffic_index + s["traffic_delta"], 0.05, 1.0))
        hours = estimate_delivery_hours(d, vehicle_type, t, weather_risk, hub_dwell_hours, driver_score)
        carbon = carbon_footprint_kg(d, vehicle_type, load_weight_kg)
        risk = sla_risk_score(hours, sla_hours, t, weather_risk, hub_dwell_hours, driver_score)

        score = 0.40 * (hours / max(sla_hours, 1)) + 0.35 * (carbon / 200) + 0.25 * risk
        results.append({
            "route": s["route"],
            "description": s["description"],
            "distance_km": d,
            "predicted_hours": hours,
            "carbon_kgco2": carbon,
            "sla_risk_score": risk,
            "risk_level": risk_label(risk),
            "decision_score": round(score, 3)
        })

    output = pd.DataFrame(results).sort_values("decision_score").reset_index(drop=True)
    output["recommendation"] = ["Recommended" if i == 0 else "Alternative" for i in range(len(output))]
    return output
