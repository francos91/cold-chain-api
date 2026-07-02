# Cold Chain Logistics Pipeline

A technical project demonstrating an end-to-end data pipeline for monitoring cold chain integrity, from data ingestion to predictive risk analysis using machine learning.

## Live Demo

The dashboard is deployed at: [https://cold-chain-api-imenmgsxnf3subrxxeqw9o.streamlit.app/](https://cold-chain-api-imenmgsxnf3subrxxeqw9o.streamlit.app/)


## Project Overview
This pipeline was developed to address the challenge of real-time logistics monitoring. It tracks shipment status and temperature excursions, providing actionable insights to identify high-risk transport corridors. The system includes a **live machine learning model** that predicts temperature breach risks for individual shipments based on historical route patterns.

### Technical Architecture
- **Data Ingestion**: A FastAPI service deployed on Render, handling REST API requests to securely ingest telemetry data.
- **Data Storage**: Supabase (PostgreSQL) serves as the primary cloud database.
- **Data Visualization & Predictive Analytics**: Streamlit dashboard that processes live data, identifies temperature excursions, and visualizes route risk scores. **Integrated with a trained Random Forest model** to provide real-time risk predictions and probability scores for each shipment.
- **Model Deployment**: Serialized model and encoder (`joblib`) stored in GitHub and loaded into the Streamlit dashboard at runtime, ensuring consistency between training and production environments.

### Repository Structure
```
cold-chain-logistics/
├── main/
│   └── main.py              # FastAPI backend (Render deployment)
├── dashboard/
│   ├── app.py               # Streamlit frontend
│   ├── advanced_model.pkl   # Trained Random Forest model
│   └── le_route.pkl         # Label encoder for route mapping
├── requirements.txt         # Python dependencies
└── README.md                # This file

```    
## Key Features
- **Predictive Risk Intelligence**: Employs a Random Forest classifier (trained on historical route data) to predict temperature excursion risks in real-time. The model identifies high-risk corridors and provides per-shipment risk probabilities, enabling proactive intervention before breaches occur.
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

## Future Roadmap
- [ ] Integrate real-time weather API data to correlate external temperature fluctuations with internal cargo excursions.
- [ ] Implement automated email/SMS alerts when a temperature breach (excursion) is detected.
- [ ] Transition from synthetic data to integration with live IoT sensor hardware.
- [ ] Integrate Carbon emissions per trip/route and possible Carbon tax cost.
- [ ] Extend the Random Forest model with XGBoost or LightGBM to improve prediction accuracy.
- [ ] Add a "What-If" simulation tool to evaluate the impact of route changes on risk scores.
- [ ] Deploy the model as a separate microservice (via Render) to allow external apps to request predictions via API.












