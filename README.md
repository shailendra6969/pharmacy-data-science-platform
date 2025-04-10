# 💊 Pharmacy Data Science Platform

A full-stack data science web application for advanced pharmaceutical analytics.

Built with Python and Streamlit, this platform integrates machine learning, natural language processing, network analysis, and time series forecasting to offer intelligent solutions in drug price prediction, medicine recommendation, sales forecasting, and gene–drug interaction visualization.

---

## 🚀 Live Demo

👉 [View Live on Streamlit Cloud](https://yourname-pharma.streamlit.app)  
*(Add your Streamlit link here once deployed)*

---

## 🎯 Features

- 🔍 **ML Models**: Predict drug prices using Random Forest, forecast sales with ARIMA
- 📈 **Sales Dashboard**: Interactive charts to monitor revenue trends and stock status
- 💬 **NLP Recommendation Engine**: Suggest similar medicines using TF-IDF + cosine similarity
- 🧬 **Gene–Drug Network**: Visualize gene interactions using NetworkX and PharmGKB API
- 🗄 **Database Integration**: Real-time SQLite + optional MongoDB support
- ⚙️ **Real-Time Updates**: Generate and refresh synthetic pharma datasets on demand
- 📊 **Unit Tested**: Includes tests for core data and model validation
- 📚 **Documentation Module**: Integrated usage guide and methodology

---

## 🛠️ Tech Stack

| Category             | Tools & Libraries                                   |
|----------------------|-----------------------------------------------------|
| Language             | Python 3.x                                           |
| UI                   | Streamlit                                            |
| ML & Forecasting     | Scikit-learn, Statsmodels (ARIMA), Random Forest    |
| NLP & Networks       | TF-IDF, CosineSimilarity, NetworkX                  |
| Visualization        | Plotly, Seaborn, Matplotlib                         |
| Database             | SQLite, MongoDB (optional)                          |
| APIs                 | PharmGKB, RxNorm, FDA (mock endpoints)              |
| DevOps               | Joblib, Logging, Unit Tests                         |

---

## 📁 Project Structure

```plaintext
pharmacy-data-science-platform/
│
├── app.py                       # Main Streamlit app
├── config.py                    # Configuration and settings
├── Pharmacy.py                  # Core processing and analytics logic
├── data/                        # Sample and generated drug datasets
├── models/                      # Trained machine learning models
├── modules/                     # Individual Streamlit UI modules
├── db/                          # Database connectors (SQLite, MongoDB)
├── utils/                       # Data loaders, generators, API helpers
├── requirement.txt              # Python dependencies
└── README.md                    # This file
