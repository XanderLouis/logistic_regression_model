# Breast Cancer Prediction

## Overview

This project is a **Breast Cancer Prediction** system built with **Streamlit and FastAPI**. It predicts whether a tumor is **benign** or **malignant** based on various medical features.

## Features

- **Interactive Web App:** Users input medical feature values to predict tumor classification.
- **Machine Learning Model:** Utilizes **Logistic Regression** for classification.
- **FastAPI Backend:** Handles model inference for predictions.
- **Streamlit Frontend:** Provides an intuitive interface.
- **Visualization:** Displays classification results with animations.

## Installation & Setup

### Prerequisites
Ensure you have **Python 3.7+** installed.

### Clone the Repository

```bash
git clone https://github.com/yourusername/breast-cancer-prediction.git
cd breast-cancer-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the FastAPI Backend

```bash
uvicorn api:app --host 0.0.0.0 --port 8001
```

### Run the Streamlit App

```bash
streamlit run app.py
```

## Usage

1. Enter the values for each medical feature.
2. Click **Predict**.
3. The app will display whether the tumor is **Benign** or **Malignant**.
4. Visualization animations are provided for engagement.

## Project Structure

```
📂 breast-cancer-prediction
├── 📂 model
│   ├── model.pkl  # Trained Logistic Regression model
│   ├── scaler.pkl  # Standard Scaler
├── 📂 data
│   ├── data_logistic_regression.csv  # Dataset
├── app.py  # Streamlit frontend
├── api.py  # FastAPI backend
├── train.py  # Model training script
├── requirements.txt  # Dependencies
└── README.md  # Project documentation
```

## Technologies Used

- **Streamlit** – For the web interface.
- **FastAPI** – For handling API requests.
- **Scikit-learn** – For Logistic Regression classification.
- **Joblib** – For model serialization.
- **Pandas & NumPy** – For data processing.
- **Uvicorn** – For running the FastAPI server.

## Contributing
Pull requests are welcome! Feel free to suggest improvements.
