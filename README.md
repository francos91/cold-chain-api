# Cold Chain Logistics Pipeline (Synthetic data used)

A technical project demonstrating an end-to-end data pipeline for monitoring cold chain integrity, from data ingestion to predictive risk analysis.

## Project Overview
This pipeline was developed to address the challenge of real-time logistics monitoring. It tracks shipment status and temperature excursions, providing actionable insights to identify high-risk transport corridors.

### Technical Architecture
- **Data Ingestion**: A FastAPI service deployed on Render, handling REST API requests to ingest telemetry data.
- **Data Storage**: Supabase (PostgreSQL) serves as the primary cloud database.
- **Data Visualization & Analytics**: A Streamlit dashboard that processes live data, identifies temperature excursions, and visualizes route risk scores using `Plotly` and `scikit-learn`.

## Key Features
- **Predictive Risk Intelligence**: Uses historical data to calculate excursion frequency per route, identifying bottlenecks where cold chain protocols are failing.
- **Scalable Data Fetching**: Implements API range/limit management to handle large datasets efficiently.
- **Interactive Filtering**: Real-time dashboard filtering based on shipment status.

## Future Roadmap
- [ ] Integrate real-time weather API data to correlate external temperature fluctuations with internal cargo excursions.
- [ ] Implement automated email/SMS alerts when a temperature breach (excursion) is detected.
- [ ] Transition from synthetic data to integration with live IoT sensor hardware.
- [ ] Integrate Carbon emissions per trip/route and possible Carbon tax cost.
