Air Quality Prediction using Deep Learning (MLP Model)

This project applies a Deep Learning (Multi-Layer Perceptron - MLP) model to predict CO(GT) (Carbon Monoxide concentration) from the Air Quality Dataset using environmental and pollutant measurements. The model aims to identify key factors affecting air quality and build a reliable regression model for pollutant prediction.

📁 Project Overview

Air pollution is a major global issue, and accurate prediction of pollutant levels helps monitor environmental health and urban air safety.
This project implements an end-to-end deep learning pipeline — from data preprocessing and visualization to model training, tuning, evaluation, and saving.

🧠 Key Features

✅ Automatic data extraction from a ZIP file
✅ Data cleaning, handling of missing and non-numeric values
✅ Exploratory Data Analysis (EDA) with rich visualizations
✅ Feature preprocessing using ColumnTransformer
✅ Neural network (MLP) built with TensorFlow / Keras
✅ Hyperparameter tuning across multiple configurations
✅ Model evaluation using RMSE, MAE, and R² metrics
✅ Visualization of performance, training curves, and residual analysis
✅ Model and preprocessing pipeline saved for future inference

🧩 Tech Stack

Python 3.x

TensorFlow / Keras

Scikit-learn

Matplotlib & Seaborn

Pandas & NumPy

Joblib

📊 Dataset

Source: Air Quality Dataset (typically from UCI / Kaggle)

Attribute	Description
CO(GT)	Carbon Monoxide concentration (ppm) — Target variable
NOx(GT), NO2(GT), C6H6(GT)	Key air pollutants
T, RH	Temperature and Relative Humidity
Other sensor readings	Environmental measurements affecting air quality

File Format: .csv (inside uploaded ZIP)

Size: ~9358 records, 15+ features

🔍 Exploratory Data Analysis

The script automatically generates the following plots:

Missing Value Chart – identifies incomplete features

Histograms – pollutant distributions

Box Plots – detect outliers

Correlation Heatmap – find inter-feature relationships

Scatter Plots – visualize pollutant and weather relationships

⚙️ Model Architecture (MLP)
Input → Dense(64, ReLU) → Dense(32, ReLU) → Dropout(0.2) → Dense(1, Linear)


Optimizer: Adam

Loss: Mean Squared Error (MSE)

Metrics: Mean Absolute Error (MAE)

Training Epochs: 50

Batch Size: 32

Hyperparameters Tested: units = [64, 128], learning_rate = [1e-3, 1e-4]

🧮 Model Evaluation
Metric	Description	Example Result
RMSE	Root Mean Squared Error	0.45
MAE	Mean Absolute Error	0.31
R²	Coefficient of Determination	0.91

The Predicted vs Actual and Residuals plots demonstrate the model’s predictive accuracy and generalization.

📈 Visualization Outputs

6️⃣ Loss (MSE) vs Epoch
7️⃣ MAE vs Epoch
8️⃣ Predicted vs Actual CO(GT)
9️⃣ Residual Distribution
🔟 Residuals vs Actual Values
1️⃣1️⃣ Metrics Summary Table

💾 Model Saving

The final trained model and preprocessing pipeline are saved for reuse:

air_quality_mlp_model.h5
air_quality_preprocessor.joblib


You can later load them for inference:

from tensorflow.keras.models import load_model
import joblib

model = load_model("air_quality_mlp_model.h5")
preprocessor = joblib.load("air_quality_preprocessor.joblib")

🚀 How to Run

1️⃣ Upload your dataset ZIP to Colab
2️⃣ Run all cells in sequence
3️⃣ View the analysis and model results in notebook output

🧾 Results Summary

The MLP model effectively learns the nonlinear relationships between temperature, humidity, and pollutant gases, yielding low RMSE and high R² values — showing strong predictive capability for CO(GT).
