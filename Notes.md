# Engineering Notes & Technical Hurdles

## 1. The "1,000 Row" API Bottleneck
**Problem**: During testing, the dashboard only displayed 1,000 records despite the database containing over 2,000 entries. 
**Diagnosis**: Both the Supabase Python client and the backend API were applying a default row limit of 1,000 as a safety feature.
**Solution**: Modified the API endpoints and the dashboard's `get_data()` function to explicitly use `.limit(5000)` and adjusted the project API settings in Supabase to permit larger payloads.

## 2. Cache Management
**Problem**: Streamlit's `st.cache_data` was keeping dashboard data stale for 10 minutes, making it difficult to verify new data ingestion.
**Solution**: Added a manual `st.cache_data.clear()` function tied to a "Refresh Data" button in the sidebar. This improved the UX and allowed for instantaneous verification during the ingestion development phase.

## 3. Data Strategy
**Note**: The dataset used in this project is synthetic. It was generated to simulate a real-world cold chain environment, including typical shipment statuses and temperature variability. This approach allowed me to focus on the architecture of the pipeline and the development of the predictive risk models without the complexity of managing proprietary, sensitive logistics data.
