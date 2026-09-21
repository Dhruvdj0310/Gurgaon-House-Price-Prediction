# 🏠 Gurgaon House Price Prediction

A machine learning based house price prediction application built with **Python, Scikit-learn, Joblib and Streamlit**.

The project trains a Random Forest regression model on housing data, saves the trained model and preprocessing pipeline, and provides a Streamlit interface for making house-value predictions.

---

## 📌 Project Overview

This project demonstrates an end-to-end machine learning workflow:

```text
Housing Dataset
      ↓
Data Preprocessing
      ↓
Train/Test Split
      ↓
Feature Transformation
      ↓
Random Forest Regressor
      ↓
Model & Pipeline Persistence
      ↓
Streamlit Prediction App
```

The application uses a saved preprocessing pipeline and trained Random Forest model to transform user inputs and generate a predicted house value.

---

## ✨ Features

- 🏠 House price/value prediction
- 📊 Numerical and categorical feature preprocessing
- 🔧 Missing-value handling using median imputation
- 📏 Feature scaling using StandardScaler
- 🔤 Categorical encoding using OneHotEncoder
- 🌲 Random Forest Regression
- 💾 Model persistence using Joblib
- 🖥️ Interactive Streamlit interface
- 📋 Property summary before prediction
- 🔍 View submitted input data
- 🎨 Custom Streamlit UI

---

## 🧠 Machine Learning Approach

The project uses a preprocessing pipeline built with Scikit-learn.

### Numerical Features

Numerical columns are processed using:

- `SimpleImputer(strategy="median")`
- `StandardScaler()`

### Categorical Features

The categorical feature:

```text
ocean_proximity
```

is processed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

The numerical and categorical transformations are combined using `ColumnTransformer`.

---

## 🌲 Machine Learning Model

The final trained model used by the application is:

```python
RandomForestRegressor(random_state=42)
```

The model is trained using the preprocessed housing features and the target:

```text
median_house_value
```

The trained model and preprocessing pipeline are saved using Joblib:

```python
joblib.dump(model, "model.pkl")
joblib.dump(pipeline, "pipeline.pkl")
```

---

## 📊 Input Features

The Streamlit application accepts the following property information:

| Feature | Description |
|---|---|
| `longitude` | Geographic longitude |
| `latitude` | Geographic latitude |
| `housing_median_age` | Median age of housing |
| `total_rooms` | Total number of rooms |
| `total_bedrooms` | Total number of bedrooms |
| `population` | Population |
| `households` | Number of households |
| `median_income` | Median income |
| `ocean_proximity` | Categorical location feature |

These inputs are converted into a Pandas DataFrame and passed through the saved preprocessing pipeline before prediction.

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit application.

The application:

1. Loads `model.pkl`
2. Loads `pipeline.pkl`
3. Accepts property details from the user
4. Converts the inputs into a DataFrame
5. Applies the saved preprocessing pipeline
6. Sends the transformed data to the Random Forest model
7. Displays the estimated house value

Run the application with:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
Gurgaon-House-Price-Prediction/
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── main.py
├── housing.csv
├── model.pkl
├── pipeline.pkl
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- Pandas
- NumPy

### Model
- Random Forest Regressor

### Data Preprocessing
- SimpleImputer
- StandardScaler
- OneHotEncoder
- ColumnTransformer
- Pipeline

### Model Persistence
- Joblib

### Application
- Streamlit

### Version Control
- Git
- GitHub
- Git LFS

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Dhruvdj0310/Gurgaon-House-Price-Prediction.git
```

### 2. Navigate to the Project

```bash
cd Gurgaon-House-Price-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 💾 Model Files

### `model.pkl`
Contains the trained Random Forest regression model.

### `pipeline.pkl`
Contains the preprocessing pipeline used to transform input features before prediction.

The application expects both files to be available in the project directory.

---

## 🔄 Prediction Pipeline

```text
User Input
    ↓
Pandas DataFrame
    ↓
Saved Preprocessing Pipeline
    ↓
Numerical Imputation
    ↓
Feature Scaling
    ↓
Categorical Encoding
    ↓
Random Forest Regressor
    ↓
Predicted House Value
```

---

## 📚 Learning Outcomes

Through this project, I worked with:

- End-to-end machine learning workflow
- Data preprocessing
- Train/test splitting
- Stratified sampling
- Feature transformation
- Scikit-learn Pipelines
- ColumnTransformer
- Numerical imputation
- Feature scaling
- One-hot encoding
- Random Forest Regression
- Model persistence with Joblib
- Streamlit application development
- Git and GitHub
- Git LFS for large model files

---

## 🚀 Future Improvements

- Add model performance metrics to the application
- Add interactive visualizations
- Add additional regression models for comparison
- Add model evaluation and cross-validation results to the UI
- Deploy the Streamlit application
- Improve input validation
- Add prediction history

---

## 👨‍💻 Author

**Dhruv Jain**

Python & AI Developer focused on Generative AI, RAG and Backend Development.

### Connect With Me

- GitHub: https://github.com/Dhruvdj0310
- LinkedIn: https://www.linkedin.com/in/dhruv-jain-0b3761258/

---

⭐ If you find this project useful, consider giving it a star!
