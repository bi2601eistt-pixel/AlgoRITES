flowchart TD
    A[GPS / IoT Data] --> E[Data Ingestion]
    B[ERP / TMS Shipment Data] --> E
    C[Traffic and Weather APIs] --> E
    D[Hub Dwell Time and Driver Data] --> E
    E --> F[Data Cleaning and Feature Engineering]
    F --> G[Delay Prediction Model]
    F --> H[Carbon Footprint Calculator]
    F --> I[SLA Risk Scoring]
    G --> J[Decision Engine]
    H --> J
    I --> J
    J --> K[Route Recommendation]
    J --> L[Dashboard and Alerts]
