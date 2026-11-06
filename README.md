# Cold Chain Logistics Transport Intelligence Dashboard
[![Cold Chain Logistics CI/CD](https://github.com/francos91/cold-chain-api/actions/workflows/ci.yml/badge.svg)](https://github.com/francos91/cold-chain-api/actions/workflows/ci.yml)

An enterprise-grade, end-to-end data pipeline and decision-support architecture for monitoring cold chain integrity, from real-time IoT telemetry ingestion to predictive risk analysis, corridor weather integration, and an autonomous AI Control Tower Copilot powered by ultra-low-latency Groq LPUs.

## Live Demos (Parallel Deployment)
- V1 (Legacy Dashboard): https://cold-chain-api-jhzb9pxulhduzxjdmcn4xt.streamlit.app/
- V2 (API-Driven Dashboard): https://cold-chain-api-xsceu8pzmbemnpx4kiqd9t.streamlit.app/
- Backend API (Docker + Render): https://cold-chain-api-docker.onrender.com

## Microservices & Docker Architecture (V2 Upgrade)
The system has been upgraded from a monolithic structure to a decoupled microservices architecture:
- Backend API (Docker + Render): The core logic is containerized using a custom Dockerfile (Python 3.11) and deployed on Render (https://cold-chain-api-docker.onrender.com/api/dashboard). This FastAPI service acts as the central data engine, loading the machine learning models (advanced_model.pkl, le_route.pkl), running feature transformations, and hosting dedicated data and telemetry routes.
- Autonomous AI Operations Copilot (Backend/agent.py): An operational agent deployed on Groq's LPU hardware (openai/gpt-oss-120b). It provides sub-2-second conversational reasoning over the /api/v1/agent/query route, utilizing function-calling across live Supabase fleet telemetry, Open-Meteo weather feeds, Random Forest model inference, and intermodal carbon tax calculators.
- Frontend Client (Overview.py & Multi-Page Suite): The V2 Streamlit dashboard acts purely as a lightweight HTTP client. It no longer stores database secrets or runs complex database queries locally. Instead, it consumes pre-processed datasets, telemetry streams, and agent endpoints directly from the Dockerized FastAPI backend.
- Data Ingestion Sanitation: Automatic normalization standardizes shipment status values (in-transit, delayed, delivered) across the API and frontend caches, eliminating duplicate entries in the filter dropdowns and analytics charts.
- Parallel Hosting: Both V1 (dashboard/app.py) and V2 (Overview.py) dashboards run concurrently on Streamlit Cloud using independent entry points within the same GitHub repository, ensuring zero downtime during architectural transitions.

## Project Overview
This pipeline was developed to address the challenge of real-time logistics monitoring. It tracks shipment status, distance, transit times, weight, and temperature excursions, providing actionable insights to identify high-risk transport corridors, evaluate financial exposure (Cost-to-Serve), and quantify carbon emissions. The system includes an autonomous AI agent and a live machine learning model that predicts temperature breach risks for individual shipments based on historical route patterns, physical distances, and transit times.

## System Architecture

### Technical Architecture
- Data Ingestion (Backend): A FastAPI service deployed on Render, handling REST API requests to ingest telemetry data and serve analytical payloads in real time.
- AI Agent Copilot (Groq LPU Engine): A dedicated reasoning layer (Backend/agent.py) using openai/gpt-oss-120b with autonomous tool calling:
  * get_active_fleet_telemetry: Direct Supabase querying to audit active fleet excursions (> 8.0°C).
  * predict_temperature_breach_risk: Containerized Random Forest inference for what-if route dispatches.
  * get_corridor_live_weather: Open-Meteo integration for ambient temperature, humidity, and Highveld pressure drops.
  * simulate_modal_shift_economics: Activity-based carbon accounting and South African Carbon Tax savings (R28.00/tCO2).
- Dual-Table Data Storage: Supabase (PostgreSQL) serves as the primary cloud database utilizing a relational dual-table structure:
  * cold_chain_tracking: The master table storing shipment records, statuses, distances, transit hours, and dynamic cargo weights.
  * temperature_stream: A time-series table storing high-frequency OSRM-generated GPS coordinates and thermodynamic temperature fluctuations.
- Frontend Dashboard (Streamlit Cloud): A multi-page control tower (pages/) that processes live data, identifies temperature excursions, quantifies carbon emissions, and visualizes route risk scores:
  * 1_Risk_Intelligence.py: Dynamic breach probability slider with partial-page rendering via @st.fragment.
  * 2_Thermal_Tracking.py: High-frequency temperature curves matched against corridor averages and GPS route maps.
  * 3_Intermodal_Simulator.py: Road vs. rail freight shift modeling under ISO 14083 emissions factors.
  * 4_AI_Copilot.py: Real-time chat interface connecting operators directly to the Groq operations agent.

## Repository Structure
```
cold-chain-api/
├── .devcontainer/
│   └── devcontainer.json        # Dev container configuration
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI/CD pipeline
├── Backend/
│   ├── agent.py                 # Autonomous GenAI Copilot (Groq LPU openai/gpt-oss-120b & native tool-calling)
│   ├── main.py                  # Backend microservice routing layer
│   └── requirements.txt         # GenAI agent & backend dependencies
├── dashboard/                   # Machine learning artifacts & legacy dashboard
├── pages/
│   ├── 1_Risk_Intelligence.py   # RF breach-risk threshold slider (@st.fragment)
│   ├── 2_Thermal_Tracking.py    # Thermal profile curves & GPS telemetry mapping
│   ├── 3_Intermodal_Simulator.py# Modal shift economics (Road vs Rail ISO 14083)
│   └── 4_AI_Copilot.py          # Conversational UI for the GenAI Operations Copilot
├── tests/                       # Unit tests (API, physics, ML, thermal)
├── Dockerfile                   # Container specification for Render deployment
├── License                      # CC BY-NC-SA 4.0 International License
├── Notes.md                     # Design notes & development logs
├── Overview.py                  # Root V2 Streamlit dashboard (SCOR KPIs)
├── README.md                    # Project documentation
├── main.py                      # FastAPI backend (Render deployment)
├── requirements.txt             # Backend / core dependencies
├── system_architecture_v3.png   # Microservices architecture specifications v3
└── system_architecture_v4.png   # GenAI Copilot & tool-calling specifications v4
```
## Two-Stage Machine Learning Architecture

This project utilizes a two-stage machine learning pipeline, separating the generation of realistic synthetic telemetry from the real-time operational dashboard.

### 1. Environmental Simulation Engine (lade_delay_model.pkl)
Because real-world cold chain telemetry is highly proprietary, this project requires a physics-based synthetic data generator. To ensure the generated data reflects real-world supply chain bottlenecks rather than random noise, the data feeder pipeline utilizes a dedicated ML model.
- Training Data: Trained on the Cainiao-AI/LaDe dataset (a real-world last-mile delivery dataset).
- Function: Predicts the probability of a shipment delay based on temporal and regional features (Time of Day, Day of Week, Area of Interest Type).
- Role in Pipeline: This model acts as the "Simulator." During the data generation phase, the Python script asked this model to predict whether each of the 4,500 simulated shipments would be delayed. If a delay was predicted, the physics engine applied a thermodynamic temperature rise. The final payload was then pushed to the FastAPI backend.

### 2. Real-Time Risk Intelligence (advanced_model.pkl)
This is the core predictive model operating inside the Control Tower. It evaluates the telemetry data sitting in the Supabase PostgreSQL database to flag high-risk shipments before a temperature excursion occurs.
- Training Data: Trained on the causal physics-based telemetry generated by the simulation engine, encompassing spatial routing, distances, transit times, and delay statuses.
- Function: A 5-feature Random Forest Classifier that predicts the continuous probability of a Cold Chain Breach (Cargo Temperature > 8.0°C).
- Role in Pipeline: Loaded directly into the FastAPI service and accessed via the frontend and Copilot tool calling. It powers the dynamic Risk Threshold Slider (optimized with @st.fragment for high-performance UI rendering), allowing operations managers to identify and intercept high-risk fleet assets based on their continuous probability scores.

## Testing & CI/CD Pipeline
This repository implements rigorous software engineering practices tailored for industrial logistics systems:
* Automated CI/CD: Powered by GitHub Actions (.github/workflows/ci.yml), running automated linting, test suites, and mock environment builds on every push.
* Unit Test Suite (tests/): 
  - test_api.py: Validates FastAPI backend response contracts, JSON schema stability, and agent endpoints.
  - test_physics.py: Verifies ISO 14083 carbon accounting, effective tax rate calculations (R28.00/tCO2), and cost-to-serve mathematics.
  - test_thermal.py: Validates thermal excursion boundary logic and thermodynamic drift thresholds for perishable cargo safety.
  - test_ml_model.py: Checks feature shape alignment and inference reliability for machine learning models.

## Key Features
- Autonomous Control Tower Copilot: Sub-2-second operational intelligence leveraging Groq LPUs (openai/gpt-oss-120b), capable of routing between natural conversation and multi-tool live telemetry audits.
- Real-Time Corridor Weather Integration: Live Open-Meteo connection tracking Karoo heatwaves and Highveld altitude-related cooling efficiency penalties (21% loss).
- Continuous Predictive Risk Intelligence: Integrates continuous physical constraints (distance_km and transit_time_hours) into the feature space alongside categorical route data to smooth the Random Forest probability distribution for granular, dynamic thresholding.
- Dynamic Risk Threshold Slider: An interactive decision-support tool (5% – 95%) leveraging @st.fragment rendering to update leaderboards instantly without full-page reloads or redundant database queries.
- Actionable High-Risk Leaderboard: Displays the highest-risk active shipments sorted by probability score. Automatically excludes already-delivered shipments, focusing operations managers strictly on active fleet assets where intervention is still possible.
- Major Corridor Transit & Distance Analysis: An executive-level data table aggregating long-haul performance across primary South African transport corridors (e.g., Johannesburg-Durban) with shipment counts, average distances, and realistic transit durations.
- Cost-to-Serve & Route Financial Exposure: Financial modeling evaluating transport cost efficiency (distance x R18.00/km + R200 fixed cost) alongside cost-vs-distance physics scatter plots with robust status color mapping (delivered, in-transit, delayed).
- Carbon Emissions & Tax Calculator: ISO 14083 / GLEC-compliant activity-based carbon accounting module that calculates tonne-kilometers (tkm) using individual shipment payload masses. Compares diesel baselines against Solar BEVs and Grid BEVs, benchmarking South African Carbon Tax liabilities at the effective rate of R28.00 per tonne of CO2 (accounting for statutory basic exemptions).

## SCOR-Aligned Performance KPIs
The Overview tab features a set of SCOR (Supply Chain Operations Reference) aligned KPIs, providing an industry-standard operational view of supply chain performance:

| KPI | SCOR Attribute | Description |
| :--- | :--- | :--- |
| Perfect Order Rate | Reliability (RL.1.1) | Percentage of shipments delivered on time and without temperature breach. |
| On-Time Delivery | Reliability (RL.2.2) | Percentage of shipments operating without delay flags. |
| Temperature Compliance | Reliability (RL.2.4) | Percentage of active shipments maintained within the 8.0°C safety threshold. |
| Average Transit Time | Responsiveness (RS.1.1) | Mean journey time (hours) derived from corridor distance and heavy-vehicle speeds. |
| Cost per Shipment | Cost (CO.3.15) | Mean trip transport cost modeled via distance (R18.00/km + fixed handling cost). |

Value at Risk (VaR), Carbon Tax Exposure, and Capacity Utilization are also calculated in the background to support Agility (AG.1.3), Cost (CO.3.15), and Asset Management (AM.3.9) metrics.

## Synthetic Data Generator & Physics Modeling
The synthetic dataset was generated using statistical parameters extracted from academic literature on South African cold chain logistics, combined with the OSRM (Open Source Routing Machine) API for realistic geodesic routing physics:

| Parameter | Literature Value / Logic | Implementation |
| :--- | :--- | :--- |
| Route Distances | Geodesic (Great Circle) * 1.25 road factor | Calculated across 10 major SA cities via OSRM |
| Transit Times | Distance / 65 km/h + dock/traffic buffers | Scaled to heavy-vehicle speeds (~12 to 35+ hours) |
| Cargo Weights | Normal distribution (12 to 30 tonnes) | Seeded dynamically per shipment in Supabase for ISO 14083 tkm |
| Mean Temperature | 0.5°C – 4.5°C | Base mean of 0.5°C (grape export studies) |
| Delay Prediction | Cainiao-AI/LaDe dataset patterns | Applied via simulated environmental model (lade_delay_model.pkl) |
| Delay Temperature Rise | +2°C to +5°C | Applied to delayed shipments |
| Altitude Penalty | ~21% cooling loss | Applied to Johannesburg/Highveld routes |
| Summer Heat Load | +2°C to +5°C | Applied probabilistically to 40% of shipments |

South Africa-Specific Literature Sources:
- Grape export reefer studies (Cape Town → Rotterdam)
- Western Cape orange cold store airlock research
- Gauteng last-mile delivery temperature studies
- Refrigeration performance at Johannesburg altitude (1,750m)

## Dataset Statistics
| Category | Count | Source / Generator |
| :--- | :--- | :--- |
| Total Master Records | 4,500 | Simulated via LaDe + Physics Engine (Active in Supabase cold_chain_tracking) |
| Time-Series Telemetry Points | 66,000 | Simulated via OSRM GPS mapping (Active in Supabase temperature_stream) |

## Future Roadmap & Completed Phases
- [x] Implement automated CI/CD pipeline and unit testing suite using GitHub Actions and pytest.
- [x] Integrate Carbon emissions per trip/route and carbon tax financial exposure modeling (aligned with South African effective rate of R28.00/tCO2).
- [x] Add an Intermodal Mode-Switching "What-If" simulation tool to evaluate the impact of shifting freight from road corridors (Natcor/Capecor) to Transnet rail.
- [x] Optimize frontend dashboard performance using @st.fragment for instantaneous widget rendering and cached temperature streams.
- [x] Resolve scatter plot status color-mapping and master shipment table pagination constraints.
- [x] Implement decoupled microservice architecture with Render Docker backend and stateless V2 Streamlit UI.
- [x] Deploy the model as a separate microservice (via Render) to allow external apps to request predictions via API.
- [x] Integrate real-time weather API (Open-Meteo) to track corridor temperatures and Karoo heat stress.
- [x] Deploy Groq LPU-powered AI Control Tower Copilot with native tool-calling across database, weather, ML, and carbon economics.
- [x] Sanitize data caching layers to eliminate multi-status duplicate values.
- [ ] Build an RL-based Smart Charging Digital Twin for electric logistics fleets under Eskom Megaflex tariffs and grid constraints (Master's Research Proposal core objective).
- [ ] Implement automated push notifications (Email/SMS webhook) for active fleet temperature excursions.
- [ ] Transition from synthetic data to integration with live IoT sensor hardware.
- [ ] Extend the Random Forest model with XGBoost or LightGBM to improve prediction accuracy.
- [ ] Add SHAP (SHapley Additive exPlanations) model interpretability to explain why specific shipments are flagged as high-risk in copilot dialogs.

## License
This project and its accompanying simulation datasets, models, and architecture are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). 

To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

Under this license:
- You are free to view, share, and adapt this work for academic, educational, or non-commercial research purposes.
- Commercial exploitation, resale, or closed-source redistribution without prior written consent from the author is strictly prohibited.
