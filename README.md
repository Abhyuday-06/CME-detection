# CME Detection & Solar Wind Analysis

This project is a comprehensive system for detecting Coronal Mass Ejections (CMEs) and predicting solar wind parameters using Machine Learning. It integrates scientific data ingestion, a deep learning prediction model, and a user-friendly web interface for researchers and viewers.

## Features

### 1. Data Ingestion & Processing
*   **SWIS-ISSDC Data Handler**: parses CDF (Common Data Format) files from the Solar Wind Ion Spectrometer (SWIS) to extract critical solar wind parameters (Proton Density, Speed, Thermal Speed).
*   **SIDC CACTus Scraper**: Automated scraper that fetches historical CME event data from the SIDC CACTus catalog for ground truth validation.
*   **PostgreSQL Integration**: Centralized storage of all timeseries and event data in a cloud-hosted NeonDB (PostgreSQL) database.

### 2. Machine Learning Model
*   **LSTM Neural Network**: A TensorFlow/Keras-based Long Short-Term Memory (LSTM) model designed for time-series forecasting.
*   **Prediction Architecture**:
    *   **Input**: 24-hour lookback window of solar parameters.
    *   **Output**: 1-hour forecast horizon for solar wind speed and density.
    *   **Training**: Includes preprocessing, scaling, and validation splits.

### 3. Web Application
*   **Role-Based Access Control**:
    *   **Scientist**: Can trigger model training, ingestion scripts, and view detailed debugging tools.
    *   **Viewer**: Read-only access to dashboards and predictions.
*   **Interactive Dashboard**: Visualizes real-time (simulated) or historical solar wind data alongside predicted CME events.
*   **Authentication**: Secure login and signup functionality.

## Data Sources

This project utilizes scientific data from the following sources:
*   **Solar Wind Data**: Sourced from **SWIS (Solar Wind Ion Spectrometer)** via the **ISSDC (Indian Space Science Data Center)**.
*   **CME Events**: Historical event catalog obtained from the **SIDC (Solar Influences Data Analysis Center)** CACTus catalog.

## Tech Stack

*   **Language**: Python 3.9+
*   **ML Framework**: TensorFlow, Keras, Scikit-learn
*   **Web Framework**: Flask
*   **Database**: PostgreSQL (psycopg2)
*   **Data Processing**: Pandas, NumPy, cdflib (for CDF files)
*   **Visualization**: Matplotlib

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r code/requirements.txt
   ```

## Usage

### Training the Model
To retrain the LSTM model on current database data:
```bash
python code/train_model.py
```

### Running the Web App
To launch the dashboard:
```bash
python code/web_app/app.py
```
Access the app at `http://127.0.0.1:5000`.

### Ingesting Data
To process new CDF files placed in the `data/` folder:
```bash
python code/feeder.py
```
