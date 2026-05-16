# GreenRoute AI - Prototype / POC

GreenRoute AI is a simple proof-of-concept for the **PT Blibli - AI-Powered Green & Resilient Logistics Network** case.

It demonstrates four core capabilities:
1. Delivery delay prediction
2. SLA risk scoring
3. Carbon footprint estimation
4. Carbon-aware route recommendation

This prototype uses synthetic logistics data for demonstration only. It is designed for student competition purposes and can be extended with real GPS, ERP, TMS, traffic, weather, and hub data.

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py` - Streamlit dashboard prototype
- `greenroute_engine.py` - AI/simulation logic
- `sample_shipments.csv` - synthetic shipment dataset
- `requirements.txt` - Python dependencies
- `poc_report_template.md` - short explanation for your proposal or demo report

## Suggested Demo Flow

1. Open the dashboard.
2. Show the synthetic shipment dataset.
3. Enter a shipment scenario in the sidebar.
4. Compare route options.
5. Explain the recommended route based on time, carbon, cost, and SLA risk.
6. Show the predicted risk level and suggested operational action.
