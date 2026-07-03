# Cold Chain Logistics Transport Intelligence Dashboard

A technical project demonstrating an end-to-end data pipeline for monitoring cold chain integrity, from data ingestion to predictive risk analysis using machine learning.

## Live Demo

The dashboard is deployed at: [https://cold-chain-api-imenmgsxnf3subrxxeqw9o.streamlit.app/](https://cold-chain-api-imenmgsxnf3subrxxeqw9o.streamlit.app/)

## Project Overview
This pipeline was developed to address the challenge of real-time logistics monitoring. It tracks shipment status and temperature excursions, providing actionable insights to identify high-risk transport corridors. The system includes a **live machine learning model** that predicts temperature breach risks for individual shipments based on historical route patterns.

## System Architecture

The following diagram illustrates the current architecture of the Cold Chain Logistics Pipeline:

![System Architecture](system_architecture_v1.png)

The system consists of:
- **Data Generation (Colab)**: Feeder script generates synthetic data using the LaDe-trained delay predictor.
- **Backend (Render)**: FastAPI serves as the data ingestion layer.
- **Database (Supabase)**: PostgreSQL stores all shipment records.
- **Frontend Dashboard (Streamlit)**: Four tabs provide analytics, risk intelligence, and thermal profiling.
- **ML Models**: Random Forest model (`advanced_model.pkl`) and label encoder (`le_route.pkl`) are loaded directly into the dashboard for real-time predictions.

### Technical Architecture
- **Data Ingestion**: A FastAPI service deployed on Render, handling REST API requests to securely ingest telemetry data.
- **Data Storage**: Supabase (PostgreSQL) serves as the primary cloud database, with separate tables for shipment records and time‑series temperature data.
- **Data Visualization & Predictive Analytics**: Streamlit dashboard that processes live data, identifies temperature excursions, and visualizes route risk scores. **Integrated with a trained Random Forest model** to provide real-time risk predictions and probability scores for each shipment.
- **Thermal Profile & GPS Analysis**: A dedicated tab allows users to visualize a shipment's 2‑hour temperature curve against the average for its status, with an interactive GPS route map colored by temperature and start/end markers.
- **Model Deployment**: Serialized model and encoder (`joblib`) stored in GitHub and loaded into the Streamlit dashboard at runtime, ensuring consistency between training and production environments.

### Repository Structure

### Repository Structure
```
cold-chain-logistics/
├── main/
│   └── main.py              # FastAPI backend (Render deployment)
├── dashboard/
│   ├── app.py               # Streamlit frontend
│   ├── advanced_model.pkl   # Trained Random Forest model
│   └── le_route.pkl         # Label encoder for route mapping
├── models/                  # ML models for delay prediction
│   ├── lade_delay_model.pkl # LaDe-trained delay predictor
│   └── lade_features.pkl    # Feature order for delay predictor
├── requirements.txt         # Python dependencies
└── README.md                # This file


```
## Key Features
- **Predictive Risk Intelligence**: Employs a Random Forest classifier (trained on historical route data) to predict temperature excursion risks in real-time. The model identifies high-risk corridors and provides per-shipment risk probabilities, enabling proactive intervention before breaches occur.
- **Dynamic Risk Threshold Slider**: An interactive decision-support tool that allows logistics managers to adjust the classification threshold (10% – 90%) based on their operational risk tolerance. Lower thresholds catch more potential breaches (higher recall) while higher thresholds reduce false alarms (higher precision).
- **Top 5 High-Risk Shipments Leaderboard**: Displays the five highest-risk active shipments sorted by probability score, allowing logistics managers to prioritize immediate action on the most critical cargo.
- **Actionable Risk Filtering**: The high-risk shipments list automatically excludes already-delivered shipments, focusing only on active shipments where intervention is still possible.
- **Thermal Profile & GPS Route Analysis**: Select any shipment to view its 2-hour temperature curve, compare it against the average for its status (delivered, delayed, in-transit, or at‑warehouse), and visualize the GPS route colored by temperature with start/end markers. This enables root‑cause analysis—identifying whether a breach occurred due to a sudden spike (door opening) or a gradual creep (refrigeration failure).
- **Live Machine Learning Integration**: The trained model and encoder are serialized using `joblib` and deployed directly within the Streamlit dashboard, delivering on-demand risk predictions for filtered datasets.
- **Interactive Filtering & Analytics**: Real-time dashboard filtering by shipment status, with dynamic updates to both historical metrics and AI-powered risk predictions.
- **Scalable Data Pipeline**: Secure data ingestion via a FastAPI backend (deployed on Render), with Supabase (PostgreSQL) as the cloud data store.

## Machine Learning Model
The predictive risk model was developed and validated in Google Colab using the following approach:

- **Feature Engineering**: Created `route_risk_score` (historical excursion frequency per route) and `is_delayed` flags to capture operational risk drivers.
- **Model**: Random Forest Classifier with `class_weight='balanced'` to handle class imbalance, achieving improved F1-score for risk prediction.
- **Feature Importance**: The model identified `route_risk_score` as the dominant predictor, confirming that spatial routing characteristics are the primary drivers of cold chain breaches.
- **Deployment**: The trained model and label encoder are serialized as `.pkl` files and deployed alongside the Streamlit dashboard, enabling live predictions without re-training.

**Performance Metrics:**
| Metric | Value |
| :--- | :--- |
| Accuracy | ~49% |
| Precision (Risk Class) | 0.43 |
| Recall (Risk Class) | 0.47 |
| F1-Score (Risk Class) | 0.45 |

## Synthetic Data Generator
The synthetic dataset was generated using statistical parameters extracted from academic literature on South African cold chain logistics. Key parameters include:

| Parameter | Literature Value | Implementation |
| :--- | :--- | :--- |
| Mean Temperature | 0.5°C – 4.5°C | Base mean of 0.5°C (grape export studies) |
| Standard Deviation | 0.6°C – 1.5°C | Base std of 0.6°C |
| Excursion Rate | 15% – 25% | 20% blended rate (TC vs NTC airlock studies) |
| Delay Temperature Rise | +2°C to +5°C | Applied to delayed shipments |
| Door Opening Rise | +3°C to +8°C | Applied to in-transit shipments |
| Altitude Penalty | ~21% cooling loss | Applied to Johannesburg/Highveld routes |
| Summer Heat Load | +2°C to +5°C | Applied probabilistically to 40% of shipments |

**South Africa-Specific Literature Sources:**
- Grape export reefer studies (Cape Town → Rotterdam)
- Western Cape orange cold store airlock research
- Gauteng last-mile delivery temperature studies
- Refrigeration performance at Johannesburg altitude (1,750m)

## Thermal Profile & GPS Route Analysis
The **Thermal Profile** tab provides an Industrial Engineering diagnostic tool that allows logistics managers to:

- **Visualize temperature curves**: See a shipment's 2‑hour temperature journey with the average curve for its status (delivered, delayed, in-transit, or at‑warehouse) for context.
- **Detect breaches**: Identify exactly when and where a shipment exceeded the 8°C threshold.
- **Analyze GPS routes**: View the shipment's route on an OpenStreetMap, colored by temperature, with start (🟢) and end (🔴) markers.
- **Compare to average**: Instantly see whether a shipment is warmer or cooler than the typical shipment with the same status.

**Example Insight**: A shipment that breaches at `0 minutes` indicates it was loaded warm at the origin—pointing to poor pre‑cooling procedures rather than in‑transit issues. This level of analysis enables targeted corrective actions.

## ML-Powered Delay Prediction (LaDe Integration)
To move beyond random assignment of delays, the system integrates a **delay prediction model trained on the LaDe dataset**—a publicly available, industrial-scale last-mile delivery dataset from China containing **4.5 million real deliveries** across multiple cities.

### How It Works:
1. **Training**: A Random Forest model was trained on 100,000 sampled rows from the LaDe dataset, using features:
   - `hour` (time of day)
   - `day_of_week` (Monday–Sunday)
   - `aoi_type` (Area of Interest: residential, commercial, industrial)
   - `time_bucket` (peak vs off-peak)
2. **Prediction**: For each new synthetic shipment, the model predicts the probability of delay.
3. **Causal Chain**: If the predicted probability exceeds 50%, the shipment is flagged as "delayed," and a temperature rise is applied based on the delay duration.

**Model Performance on LaDe Data:**
| Metric | Value |
| :--- | :--- |
| Accuracy | 64.8% |
| Key Feature | `aoi_type` (42% importance) |
| Key Feature | `hour` (33% importance) |
| Delay Rate | 36.7% (realistic for urban delivery) |

**Why This Matters**: By training on real-world logistics data, the delay predictor adds **empirical grounding** to the synthetic data pipeline, making the simulated delays more realistic and defensible for industrial engineering research.

## Decision-Support Tools
The dashboard includes an interactive **Risk Threshold Slider** that allows users to:
- **Adjust risk sensitivity**: Slide between 10% and 90% to control how many shipments are flagged as high-risk.
- **Balance precision vs. recall**: Lower thresholds catch more actual breaches but may increase false alarms.
- **Align with business needs**: Match the model's behavior to operational risk tolerance and financial constraints.
- **See immediate feedback**: The "Predicted High-Risk Shipments" count and the high-risk lists update instantly as the slider is moved.

## Dataset Statistics
| Category | Count | Source |
| :--- | :--- | :--- |
| Synthetic (CAUSAL-) | 4,500 | South African cold chain literature |
| ML-Powered (ML-LADE-) | 1,000 | LaDe-trained delay predictor |
| **Total** | **5,500** | Complete dataset |

**Validation**: At a 25% risk threshold, the model correctly identifies **all 1,228 delayed shipments** as high-risk, confirming the causal link between delays and temperature breaches.

## Future Roadmap
- [ ] Integrate real-time weather API data to correlate external temperature fluctuations with internal cargo excursions.
- [ ] Implement automated email/SMS alerts when a temperature breach (excursion) is detected.
- [ ] Transition from synthetic data to integration with live IoT sensor hardware.
- [ ] Integrate Carbon emissions per trip/route and possible Carbon tax cost.
- [ ] Extend the Random Forest model with XGBoost or LightGBM to improve prediction accuracy.
- [ ] Add a "What-If" simulation tool to evaluate the impact of route changes on risk scores.
- [ ] Deploy the model as a separate microservice (via Render) to allow external apps to request predictions via API.
- [ ] Add SHAP (SHapley Additive exPlanations) model interpretability to explain why specific shipments are flagged as high-risk.
- [ ] Add a "Similar Shipments" view to group shipments with matching temperature profiles.

## Notes
- This project uses synthetic data to simulate real-world cold chain conditions. The system is designed to accept live IoT sensor data with zero code changes.
- The dashboard includes pagination to handle large datasets efficiently, ensuring scalability as the database grows.
- The synthetic data generator is grounded in South African cold chain literature, making it suitable for academic research and industrial engineering applications.
- **The LaDe delay predictor** adds empirical real-world grounding to the delay simulation, trained on 4.5 million real deliveries from urban China.
- **The Thermal Profile tab** uses real GPS routes generated via OpenStreetMap (OSRM) and includes time‑series temperature data for each shipment, enabling precise root‑cause analysis.

   
