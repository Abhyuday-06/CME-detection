# CME Detection & Solar Wind Analysis

A comprehensive system for detecting Coronal Mass Ejections (CMEs) and predicting solar wind parameters using Machine Learning. It integrates multi-source scientific data ingestion, deep learning model comparison, exploratory data analysis, and a role-based web interface for researchers and viewers.

## Features

### 1. Data Ingestion & Processing
*   **SWIS-ISSDC Data Handler** (`feeder.py`, `spooler.py`): Parses CDF (Common Data Format) files from the Solar Wind Ion Spectrometer (SWIS) aboard Aditya-L1 to extract proton density, speed, thermal speed, and alpha particle parameters. Supports both bulk historical backfill and real-time simulation.
*   **SIDC CACTus Scraper** (`cactus_scraper.py`): Automated scraper that fetches historical CME event data from the SIDC CACTus catalog for ground truth labeling.
*   **DSCOVR Plasma Fetcher** (`fetch_dscovr_plasma.py`): Retrieves NOAA DSCOVR L1 solar wind plasma data (speed, density, temperature).
*   **NASA OMNI2 Fetcher** (`fetch_omni_historical.py`): Downloads hourly solar wind and Kp index data from NASA OMNIWeb covering the Aditya-L1 observation period.
*   **Kp Index Fetchers** (`fetch_kp_index.py`, `fetch_kp_historical.py`): Ingest NOAA planetary Kp geomagnetic index data for correlation analysis.
*   **PostgreSQL Integration**: Centralized storage of all timeseries and event data in a cloud-hosted NeonDB (PostgreSQL) database. Schema utilities (`apply_schema_updates.py`, `update_schema_v2.py`) maintain versioned migrations.

### 2. Exploratory Data Analysis (`eda.py`)
Produces 10 publication-quality plots saved to `code/eda_output/`:
*   Aditya-L1, DSCOVR, and Kp index time-series overviews
*   Distribution plots and correlation matrix across all three data sources
*   Alpha particle ratio analysis, data quality heatmap, timestamp overlap heatmap, and outlier boxplots

### 3. Machine Learning Models
*   **LSTM Neural Network** (`train_model.py`): TensorFlow/Keras LSTM designed for solar wind time-series forecasting.
    *   **Input**: 24-hour lookback window of solar parameters.
    *   **Output**: 1-hour ahead Kp index forecast.
*   **Model Comparison** (`model_comparison.py`): Side-by-side benchmarking of three architectures on the same dataset:
    *   **LSTM** (TensorFlow/Keras)
    *   **PINN** — Physics-Informed Neural Network (PyTorch)
    *   **Quantum-Inspired Variational Model** (PyTorch)
    *   Results (RMSE, MAE, R²) persisted to `code/model_output/results.json` with comparison plots.

### 4. Web Application (`web_app/`)
*   **Role-Based Access Control**:
    *   **Scientist**: Can trigger model training, data ingestion scripts, and access diagnostic tools.
    *   **Viewer**: Read-only access to dashboards and predictions.
*   **Interactive Dashboard**: Visualizes historical solar wind data alongside model-predicted CME events.
*   **Authentication**: Secure login and signup with hashed credentials.

## Database Schema

![ER Diagram](submission_resources/ER_Diagram.png)
![Schema Diagram](submission_resources/Schema_Diagram.png)

## Data Sources

| Source | Data | Script |
|---|---|---|
| ISRO Aditya-L1 ASPEX/SWIS | Solar wind plasma (CDF) | `feeder.py`, `spooler.py` |
| NOAA DSCOVR | L1 plasma (speed, density, temp) | `fetch_dscovr_plasma.py` |
| NASA OMNIWeb (OMNI2) | Hourly solar wind + Kp index | `fetch_omni_historical.py` |
| NOAA GFZ | Planetary Kp index | `fetch_kp_index.py`, `fetch_kp_historical.py` |
| SIDC CACTus | Historical CME event catalog | `cactus_scraper.py` |

## Tech Stack

*   **Language**: Python 3.9+
*   **ML Frameworks**: TensorFlow / Keras, PyTorch, Scikit-learn
*   **Web Framework**: Flask, Flask-Login
*   **Database**: PostgreSQL (NeonDB) via psycopg2
*   **Data Processing**: Pandas, NumPy, cdflib (CDF files), Requests
*   **Visualization**: Matplotlib

## Project Structure

```
CME-Detection/
├── code/
│   ├── feeder.py                  # Bulk CDF → DB ingestion (Aditya-L1)
│   ├── spooler.py                 # Real-time CDF simulation feed
│   ├── cactus_scraper.py          # SIDC CACTus CME event scraper
│   ├── fetch_dscovr_plasma.py     # NOAA DSCOVR plasma fetcher
│   ├── fetch_omni_historical.py   # NASA OMNI2 historical fetcher
│   ├── fetch_kp_index.py          # NOAA Kp index fetcher
│   ├── fetch_kp_historical.py     # Historical Kp backfill
│   ├── eda.py                     # Exploratory Data Analysis & plots
│   ├── train_model.py             # LSTM model training
│   ├── model_comparison.py        # LSTM vs PINN vs Quantum-Inspired
│   ├── detection.py               # CME detection logic
│   ├── visualizer.py              # Visualization utilities
│   ├── apply_schema_updates.py    # DB schema migration v1
│   ├── update_schema_v2.py        # DB schema migration v2
│   ├── eda_output/                # EDA plots (PNG)
│   ├── model_output/              # Model comparison plots + results.json
│   └── web_app/
│       ├── app.py                 # Flask application
│       ├── static/                # Static assets (icons, CSS)
│       └── templates/             # Jinja2 HTML templates
├── submission_resources/
│   ├── ER_Diagram.png
│   └── Schema_Diagram.png
├── requirements.txt → code/requirements.txt
└── README.md
```

## Setup

1.  Create a virtual environment:
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

2.  Install dependencies:
    ```bash
    pip install -r code/requirements.txt
    ```

3.  Create a `.env` file in the project root with your database connection string:
    ```
    DB_URI=postgresql://user:password@host/dbname
    ```

## Usage

### Training the Model
```bash
python code/train_model.py
```

### Running Model Comparison (LSTM vs PINN vs Quantum-Inspired)
```bash
python code/model_comparison.py
# Outputs plots and results.json to code/model_output/
```

### Running Exploratory Data Analysis
```bash
python code/eda.py
# Outputs 10 plots to code/eda_output/
```

### Ingesting Data

```bash
# Bulk ingest Aditya-L1 CDF files from data/ folder
python code/feeder.py

# Real-time simulation feed
python code/spooler.py

# Fetch DSCOVR plasma data
python code/fetch_dscovr_plasma.py

# Fetch NASA OMNI2 data
python code/fetch_omni_historical.py

# Fetch Kp index
python code/fetch_kp_index.py
```

### Running the Web App
```bash
python code/web_app/app.py
```
Access the app at `http://127.0.0.1:5000`.
