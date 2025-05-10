

# Machine Condition Prediction using Random Forest

**By Abishek V**
**2nd Year – Mechanical Engineering**
**ARM College of Engineering & Technology**
**Course: Data Analysis in Mechanical Engineering**

---

## About the Project

This project is focused on predicting the condition of a machine using a *Random Forest Classifier*. The idea is to analyze input data such as temperature, vibration, oil quality, RPM, and similar mechanical indicators to find out if the machine is running normally or if there might be a fault.

The aim is to make maintenance smarter by using data-driven insights.

---

## Getting Started

Before running the code, make sure you have all the required Python packages installed. You can do that by running:

```bash
pip install -r requirements.txt
```

---

## Important Files

Here are the key files needed for running predictions:

* `random_forest_model.pkl` – This is the pre-trained Random Forest model.
* `scaler.pkl` – A StandardScaler from Scikit-learn used to normalize inputs before prediction.
* `selected_features.pkl` – A list of feature names that the model expects, to maintain correct order.

All of these files should be placed in the same directory where your prediction script runs, unless paths are specified differently.

---

## How the Prediction Works

1. **Loading the Necessary Files:**

   We load the saved model, the scaler, and the list of selected features using `joblib.load()`.

2. **Input Preparation:**

   Prepare a single row of input data using a `pandas.DataFrame`. This data should include all the required features in the correct order.

3. **Preprocessing:**

   The input is scaled using the loaded `scaler` so that it matches the format of the data used during training.

4. **Prediction:**

   Use `.predict()` to find out the predicted machine condition, and `.predict_proba()` to understand the confidence levels of the prediction.

---

## Sample Code for Prediction

Here is a basic script that shows how to use the model for prediction:

```python
import joblib
import pandas as pd

# Load the model and preprocessing files
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

# Example input (replace with real values during actual use)
new_data = pd.DataFrame([{
    'Temperature_C': 75,
    'Vibration_mm_s': 2.5,
    'Oil_Quality_Index': 88,
    'RPM': 1500,
    'Pressure_bar': 5.2,
    'Shaft_Misalignment_deg': 0.3,
    'Noise_dB': 70,
    'Load_%': 85,
    'Power_kW': 12.5
}])

# Arrange features in the correct order
new_data = new_data[selected_features]

# Scale the data
scaled_data = scaler.transform(new_data)

# Make predictions
prediction = model.predict(scaled_data)
prediction_proba = model.predict_proba(scaled_data)

print("Predicted Class:", prediction[0])
print("Prediction Probabilities:", prediction_proba[0])
```

---

## Things to Keep in Mind

* Your input should include **exactly** the same features as used in training.
* Keep the feature values realistic and similar to what the model was trained on.
* Never change the order of columns unless it matches the `selected_features.pkl`.

---

## Retraining the Model (Optional)

If you want to retrain or improve the model:

* Use the same preprocessing methods.
* Apply consistent feature selection and scaling.
* Save the updated files using `joblib` just like in the original setup.

ection in rotating machinery.
* IoT-based machine monitoring systems.

---

Let me know if you'd like a PDF version or diagram to go with the README. Would you like to include a short personal introduction or motivation for why you chose this topic?
