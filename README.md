# 🌍 Soil UCS Prediction Using ANN

A Streamlit-based machine learning application for predicting the **Unconfined Compressive Strength (UCS)** of clayey soils from their index properties.

The application integrates:

- 🧠 Artificial Neural Network (ANN)
- 🔄 5-Fold Cross-Validation
- 📊 Model Performance Evaluation
- 🔍 Explainable Artificial Intelligence (SHAP)
- 🤖 Gemini AI-based Engineering Interpretation
- 🌐 Streamlit Web Application

---

## 📌 Research Objective

The objective of this application is to predict the **Unconfined Compressive Strength (UCS)** of clayey soils using easily determined soil index properties.

The application supports research on the development of predictive models for engineering behavior of clayey soils using experimental data and machine learning techniques.

---

# 🏗️ Application Workflow

```text
Excel Soil Dataset
        │
        ▼
Data Loading and Feature Selection
        │
        ▼
Data Preprocessing
(Imputation + Standardization)
        │
        ▼
Artificial Neural Network
        │
        ▼
5-Fold Cross-Validation
        │
        ├───────────────┐
        ▼               ▼
Model Performance    Explainable AI
Evaluation              (SHAP)
        │               │
        └───────┬───────┘
                ▼
          Final ANN Model
                │
                ▼
          UCS Prediction
                │
                ▼
     Gemini AI Interpretation
```

---

# 📊 Input Parameters

The application is designed to predict UCS using soil index properties.

The default input parameters are:

| No. | Parameter | Symbol |
|---|---|---|
| 1 | Liquid Limit | LL (%) |
| 2 | Plastic Limit | PL (%) |
| 3 | Plasticity Index | PI (%) |
| 4 | Clay Fraction | Clay Fraction (%) |
| 5 | Clay Activity | Clay Activity (-) |
| 6 | Water Content | w (%) |
| 7 | Specific Gravity | Gs (-) |

The user can also select different input parameters available in the uploaded dataset.

---

# 🎯 Output Parameter

The target output parameter is:

```text
UCS (kPa)
```

**UCS = Unconfined Compressive Strength**

---

# 🧠 Artificial Neural Network Model

The application uses an ANN regression model implemented using:

```text
MLPRegressor
```

The default network architecture is:

```text
Input Layer
     │
     ▼
Hidden Layer 1
64 Neurons
     │
     ▼
Hidden Layer 2
32 Neurons
     │
     ▼
Output Layer
1 Neuron (UCS)
```

### Model Settings

- Activation Function: ReLU
- Solver/Optimizer: Adam
- Feature Scaling: StandardScaler
- Missing Value Treatment: Median Imputation
- Maximum Iterations: 2000
- Early Stopping: Enabled

The ANN architecture can be modified from the Streamlit sidebar.

---

# 🔄 5-Fold Cross-Validation

The model uses **K-Fold Cross-Validation with K = 5**.

The dataset is divided into five parts.

```text
Fold 1 → Train on 4 parts → Test on 1 part
Fold 2 → Train on 4 parts → Test on 1 part
Fold 3 → Train on 4 parts → Test on 1 part
Fold 4 → Train on 4 parts → Test on 1 part
Fold 5 → Train on 4 parts → Test on 1 part
```

The average performance across all folds is then calculated.

---

# 📈 Performance Metrics

The following evaluation metrics are calculated.

## 1. Coefficient of Determination (R²)

Measures how well the predicted UCS values match the actual values.

```text
Higher R² indicates better model performance.
```

## 2. Mean Absolute Error (MAE)

Measures the average magnitude of prediction errors.

```text
Lower MAE indicates better performance.
```

## 3. Root Mean Squared Error (RMSE)

Gives greater importance to large prediction errors.

```text
Lower RMSE indicates better performance.
```

## 4. Mean Absolute Percentage Error (MAPE)

Measures the prediction error as a percentage.

```text
Lower MAPE indicates better performance.
```

---

# 🔍 Explainable Artificial Intelligence (XAI)

The application uses **SHAP (SHapley Additive exPlanations)** to interpret the ANN model.

SHAP analysis provides:

- Global feature importance
- Mean absolute SHAP values
- SHAP summary plots
- Understanding of the contribution of each input feature

This helps identify which soil index properties have the greatest influence on UCS prediction.

---

# ⚠️ Important Research Note

Plasticity Index is calculated as:

```text
PI = LL − PL
```

Therefore:

- LL
- PL
- PI

are mathematically related.

Including all three parameters may introduce multicollinearity or feature redundancy. Therefore, feature importance results should be interpreted carefully.

Future versions of the model may compare different feature combinations.

---

# 🤖 Gemini AI Integration

The application integrates the Gemini API for **engineering-language interpretation**.

### Important

Gemini does **not** perform the numerical UCS prediction.

The workflow is:

```text
Soil Input Parameters
        │
        ▼
ANN Model
        │
        ▼
Numerical UCS Prediction
        │
        ▼
Gemini AI
        │
        ▼
Engineering Interpretation
```

Gemini helps explain the predicted UCS in academic and engineering language.

---

# 📁 Project Structure

```text
Soil-UCS-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Soil-UCS-Prediction.git
```

## 2. Navigate to the Project Folder

```bash
cd Soil-UCS-Prediction
```

## 3. Install Required Packages

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application Locally

Run the following command:

```bash
streamlit run app.py
```

The application will open in your web browser.

---

# 🌐 Deploy on Streamlit Community Cloud

## Step 1: Create a GitHub Repository

Create a repository named:

```text
Soil-UCS-Prediction
```

Upload the following files:

```text
app.py
requirements.txt
README.md
```

---

## Step 2: Connect GitHub to Streamlit

Go to Streamlit Community Cloud.

Select:

```text
Create App
```

Then select:

- GitHub Repository
- Branch: `main`
- Main file path: `app.py`

---

## Step 3: Deploy

Click:

```text
Deploy
```

Streamlit will automatically install the packages from:

```text
requirements.txt
```

---

# 🔐 Gemini API Key Setup

Do not place your Gemini API key directly inside:

```text
app.py
```

Do not upload your API key to GitHub.

Instead, add the API key in Streamlit Secrets.

Use:

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```

The application reads the API key securely using:

```python
st.secrets["GEMINI_API_KEY"]
```

---

# 📄 Dataset Requirements

The dataset should be in:

```text
.xlsx
```

or:

```text
.xls
```

format.

The dataset should contain numerical values for the input parameters and UCS target.

Example:

| LL (%) | PL (%) | PI (%) | Clay Fraction (%) | Clay Activity (-) | w (%) | Gs (-) | UCS (kPa) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 45 | 22 | 23 | 35 | 0.66 | 18 | 2.70 | 180 |
| 52 | 25 | 27 | 42 | 0.64 | 20 | 2.72 | 210 |

---

# 🖥️ Application Features

The Streamlit application contains six main sections.

## 📁 1. Dataset

- Upload Excel dataset
- Preview data
- Check number of samples
- Check number of columns
- Check missing values
- Select input features
- Select target output

---

## 🧠 2. ANN and 5-Fold Cross-Validation

- Configure ANN architecture
- Train ANN model
- Perform 5-fold cross-validation
- View performance for each fold
- View average performance

---

## 📊 3. Model Performance

- Overall R²
- MAE
- RMSE
- MAPE
- Actual vs Predicted UCS plot
- Residual plot
- Download prediction results

---

## 🔍 4. Explainable AI

- SHAP feature importance
- Mean absolute SHAP values
- Global feature importance plot
- SHAP summary plot

---

## 🔮 5. Predict UCS

Users can enter soil index properties manually.

Example:

```text
LL = 45 %
PL = 22 %
PI = 23 %
Clay Fraction = 35 %
Clay Activity = 0.66
Water Content = 18 %
Specific Gravity = 2.70
```

The ANN model then predicts:

```text
Predicted UCS = XXX.XX kPa
```

---

## 🤖 6. Gemini Interpretation

Gemini AI provides a concise engineering interpretation of the ANN prediction.

The interpretation considers:

- Soil plasticity
- Clay fraction
- Clay activity
- Water content
- Specific gravity
- Feature relationships

---

# 📦 Required Libraries

The application requires:

- Streamlit
- Pandas
- NumPy
- Scikit-learn
- OpenPyXL
- Matplotlib
- SHAP
- Google GenAI

All dependencies are included in:

```text
requirements.txt
```

---

# 🔬 Research Methodology

The overall methodology is:

```text
Soil Sample Data
        ↓
Laboratory Testing
        ↓
Determination of Index Properties
        ↓
Dataset Development
        ↓
Data Preprocessing
        ↓
Feature Selection
        ↓
ANN Model Development
        ↓
5-Fold Cross-Validation
        ↓
Performance Evaluation
        ↓
Explainable AI (SHAP)
        ↓
UCS Prediction Application
```

---

# 🎓 Research Application

This application supports research related to:

> **Development of Predictive Models for Shear Strength, CBR and Compaction of Clayey Soils Using Index Properties and Machine Learning**

The present application specifically focuses on:

> **Prediction of Unconfined Compressive Strength (UCS) of Clayey Soils Using Index Properties and Artificial Neural Networks**

---

# 🚀 Future Improvements

Possible future developments include:

- Comparison of ANN with Random Forest
- Support Vector Regression
- XGBoost
- Ensemble learning
- Hyperparameter optimization
- Automated feature selection
- Prediction intervals
- Multi-output prediction
- Model saving and loading
- Batch prediction using Excel files
- PDF report generation
- LIME-based explanations
- Comparison of different ANN architectures

---

# 👨‍🔬 Author

**Izat Ali Sahito**

PhD Researcher  
Civil Engineering / Geotechnical Engineering

Research Area:

**Machine Learning Applications in Geotechnical Engineering**

---

# 📜 License

This project is intended for academic and research purposes.

---

## ⭐ Acknowledgement

This application combines geotechnical engineering principles, laboratory soil testing data, machine learning, explainable artificial intelligence, and web-based deployment to support efficient prediction and interpretation of soil engineering properties.
