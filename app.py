import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from google import genai
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Soil UCS Prediction",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Soil UCS Prediction Using ANN")
st.caption(
    "Prediction of Unconfined Compressive Strength (UCS) from Soil Index Properties "
    "using Artificial Neural Networks, 5-Fold Cross-Validation and Explainable AI."
)

# ============================================================
# DEFAULT FEATURES
# ============================================================
DEFAULT_FEATURES = [
    "LL (%)",
    "PL (%)",
    "PI (%)",
    "Clay Fraction (%)",
    "Clay Activity (-)",
    "w (%)",
    "Gs (-)",
]

TARGET_NAME = "UCS (kPa)"


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def clean_column_names(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def calculate_metrics(y_true, y_pred):
    return {
        "R²": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


@st.cache_data(show_spinner=False)
def load_excel(file):
    return clean_column_names(pd.read_excel(file))


def build_ann(random_state=42):
    """
    ANN regression model using sklearn MLPRegressor.

    Pipeline:
    Missing-value imputation -> Standardization -> ANN
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "ann",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    learning_rate_init=0.001,
                    max_iter=2000,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=50,
                    random_state=random_state,
                ),
            ),
        ]
    )


def get_gemini_key():
    """Read Gemini API key safely from Streamlit Secrets or environment."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY")


def gemini_interpretation(api_key, features_dict, predicted_ucs, feature_names):
    """
    Uses Gemini only for engineering-language interpretation.
    The numerical prediction always comes from the ANN model.
    """
    if not api_key:
        return (
            "Gemini API key was not found. Add GEMINI_API_KEY to "
            "Streamlit Secrets to enable AI interpretation."
        )

    prompt = f"""
You are an assistant for geotechnical engineering research.

A machine-learning ANN model predicted the Unconfined Compressive Strength
(UCS) of a clayey soil.

Input soil properties:
{features_dict}

ANN predicted UCS:
{predicted_ucs:.2f} kPa

Available model input features:
{", ".join(feature_names)}

Write a concise engineering interpretation in 120-180 words.
Important:
1. Do not claim that Gemini performed the numerical prediction.
2. State that the ANN model produced the UCS value.
3. Explain generally how plasticity, clay fraction, clay activity,
   water content and specific gravity can influence soil strength.
4. Do not present unsupported causal conclusions.
5. Mention that LL, PL and PI are mathematically related, so their
   individual effects should be interpreted carefully.
6. Use clear academic language suitable for a PhD research application.
"""

    client = genai.Client(api_key=api_key)

    # A current Gemini model name can be changed here if required by API availability.
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Model Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload Soil Dataset (Excel)",
    type=["xlsx", "xls"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("ANN Configuration")

hidden_layer_1 = st.sidebar.number_input(
    "Hidden Layer 1 Neurons",
    min_value=4,
    max_value=256,
    value=64,
    step=4,
)

hidden_layer_2 = st.sidebar.number_input(
    "Hidden Layer 2 Neurons",
    min_value=4,
    max_value=256,
    value=32,
    step=4,
)

max_iter = st.sidebar.number_input(
    "Maximum Iterations",
    min_value=200,
    max_value=5000,
    value=2000,
    step=100,
)

random_state = st.sidebar.number_input(
    "Random State",
    min_value=0,
    max_value=9999,
    value=42,
    step=1,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "For security, do not place the Gemini API key directly in app.py. "
    "Use Streamlit Secrets."
)


# ============================================================
# DATA LOADING
# ============================================================
if uploaded_file is None:
    st.info("👈 Please upload your Excel soil dataset to begin.")

    st.markdown("### Expected Input Features")
    st.write(DEFAULT_FEATURES)

    st.markdown("### Expected Target")
    st.write(TARGET_NAME)

    st.stop()

try:
    data = load_excel(uploaded_file)
except Exception as e:
    st.error(f"Unable to read the Excel file: {e}")
    st.stop()

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📁 Dataset",
        "🧠 ANN & 5-Fold CV",
        "📊 Performance",
        "🔍 Explainable AI",
        "🔮 Predict UCS",
        "🤖 Gemini Interpretation",
    ]
)


# ============================================================
# TAB 1: DATASET
# ============================================================
with tab1:
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Samples", data.shape[0])
    col2.metric("Number of Columns", data.shape[1])
    col3.metric("Missing Values", int(data.isnull().sum().sum()))

    st.dataframe(data.head(10), use_container_width=True)

    st.subheader("Column Selection")

    all_columns = list(data.columns)

    default_available = [
        feature for feature in DEFAULT_FEATURES if feature in all_columns
    ]

    selected_features = st.multiselect(
        "Select Input Features",
        options=all_columns,
        default=default_available,
        key="selected_features",
    )

    default_target_index = (
        all_columns.index(TARGET_NAME)
        if TARGET_NAME in all_columns
        else len(all_columns) - 1
    )

    target = st.selectbox(
        "Select Target Output",
        options=all_columns,
        index=default_target_index,
        key="target_column",
    )

    if "S.No." in selected_features:
        st.warning("S.No. is an identification column and should normally be excluded.")

    if target in selected_features:
        st.error("The target column cannot also be an input feature.")

    st.subheader("Descriptive Statistics")
    st.dataframe(data.describe(include="all").T, use_container_width=True)


# ============================================================
# MODEL PREPARATION
# ============================================================
selected_features = st.session_state.get(
    "selected_features",
    default_available,
)
target = st.session_state.get(
    "target_column",
    TARGET_NAME if TARGET_NAME in all_columns else all_columns[-1],
)

valid_setup = (
    len(selected_features) >= 2
    and target in data.columns
    and target not in selected_features
)

if not valid_setup:
    st.error(
        "Please select at least two input features and a different target output."
    )
    st.stop()

model_data = data[selected_features + [target]].copy()

for column in selected_features + [target]:
    model_data[column] = pd.to_numeric(model_data[column], errors="coerce")

model_data = model_data.dropna(subset=[target])

X = model_data[selected_features]
y = model_data[target]

# Create configurable ANN pipeline
model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "ann",
            MLPRegressor(
                hidden_layer_sizes=(int(hidden_layer_1), int(hidden_layer_2)),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=int(max_iter),
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=50,
                random_state=int(random_state),
            ),
        ),
    ]
)

# ============================================================
# TAB 2: 5-FOLD CROSS-VALIDATION
# ============================================================
with tab2:
    st.subheader("ANN Model and 5-Fold Cross-Validation")

    st.markdown(
        f"""
**Model Architecture**

- Input neurons: **{len(selected_features)}**
- Hidden Layer 1: **{int(hidden_layer_1)} neurons**
- Hidden Layer 2: **{int(hidden_layer_2)} neurons**
- Output neurons: **1 (UCS)**
- Activation function: **ReLU**
- Optimizer: **Adam**
- Validation method: **5-Fold Cross-Validation**
"""
    )

    train_button = st.button("🚀 Train ANN Model", type="primary")

    if train_button or "trained_results" not in st.session_state:
        with st.spinner("Training ANN model using 5-fold cross-validation..."):
            kfold = KFold(
                n_splits=5,
                shuffle=True,
                random_state=int(random_state),
            )

            fold_rows = []
            all_predictions = np.zeros(len(y))

            for fold, (train_idx, test_idx) in enumerate(
                kfold.split(X), start=1
            ):
                fold_model = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        (
                            "ann",
                            MLPRegressor(
                                hidden_layer_sizes=(
                                    int(hidden_layer_1),
                                    int(hidden_layer_2),
                                ),
                                activation="relu",
                                solver="adam",
                                alpha=0.001,
                                learning_rate_init=0.001,
                                max_iter=int(max_iter),
                                early_stopping=True,
                                validation_fraction=0.15,
                                n_iter_no_change=50,
                                random_state=int(random_state) + fold,
                            ),
                        ),
                    ]
                )

                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                fold_model.fit(X_train, y_train)
                fold_pred = fold_model.predict(X_test)

                all_predictions[test_idx] = fold_pred

                metrics = calculate_metrics(y_test, fold_pred)
                metrics["Fold"] = fold

                fold_rows.append(metrics)

            fold_results = pd.DataFrame(fold_rows)

            overall_metrics = calculate_metrics(y, all_predictions)

            # Final model trained on all available data
            final_model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "ann",
                        MLPRegressor(
                            hidden_layer_sizes=(
                                int(hidden_layer_1),
                                int(hidden_layer_2),
                            ),
                            activation="relu",
                            solver="adam",
                            alpha=0.001,
                            learning_rate_init=0.001,
                            max_iter=int(max_iter),
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=50,
                            random_state=int(random_state),
                        ),
                    ),
                ]
            )

            final_model.fit(X, y)

            st.session_state["trained_results"] = {
                "fold_results": fold_results,
                "overall_metrics": overall_metrics,
                "predictions": all_predictions,
                "actual": y.values,
                "final_model": final_model,
                "features": selected_features,
                "target": target,
                "X": X.copy(),
            }

    if "trained_results" in st.session_state:
        results = st.session_state["trained_results"]

        st.success("ANN model training completed successfully.")

        st.subheader("Cross-Validation Results")
        display_fold = results["fold_results"][
            ["Fold", "R²", "MAE", "RMSE", "MAPE (%)"]
        ].copy()

        st.dataframe(
            display_fold.style.format(
                {
                    "R²": "{:.4f}",
                    "MAE": "{:.4f}",
                    "RMSE": "{:.4f}",
                    "MAPE (%)": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        st.subheader("Average 5-Fold Performance")

        average_metrics = results["fold_results"][
            ["R²", "MAE", "RMSE", "MAPE (%)"]
        ].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean R²", f"{average_metrics['R²']:.4f}")
        c2.metric("Mean MAE", f"{average_metrics['MAE']:.4f}")
        c3.metric("Mean RMSE", f"{average_metrics['RMSE']:.4f}")
        c4.metric("Mean MAPE", f"{average_metrics['MAPE (%)']:.2f}%")


# ============================================================
# TAB 3: PERFORMANCE
# ============================================================
with tab3:
    st.subheader("Model Performance")

    if "trained_results" not in st.session_state:
        st.info("Please train the ANN model first.")
    else:
        results = st.session_state["trained_results"]

        y_actual = results["actual"]
        y_pred = results["predictions"]

        overall = results["overall_metrics"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall R²", f"{overall['R²']:.4f}")
        c2.metric("Overall MAE", f"{overall['MAE']:.4f}")
        c3.metric("Overall RMSE", f"{overall['RMSE']:.4f}")
        c4.metric("Overall MAPE", f"{overall['MAPE (%)']:.2f}%")

        st.subheader("Actual vs Predicted UCS")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_actual, y_pred, alpha=0.75)

        min_value = min(np.min(y_actual), np.min(y_pred))
        max_value = max(np.max(y_actual), np.max(y_pred))

        ax.plot(
            [min_value, max_value],
            [min_value, max_value],
            linestyle="--",
        )

        ax.set_xlabel("Actual UCS (kPa)")
        ax.set_ylabel("Predicted UCS (kPa)")
        ax.set_title("Actual vs Predicted UCS")
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

        st.subheader("Residual Plot")

        residuals = y_actual - y_pred

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.scatter(y_pred, residuals, alpha=0.75)
        ax2.axhline(0, linestyle="--")

        ax2.set_xlabel("Predicted UCS (kPa)")
        ax2.set_ylabel("Residual (Actual - Predicted)")
        ax2.set_title("Residual Plot")
        ax2.grid(True, alpha=0.3)

        st.pyplot(fig2)

        prediction_table = pd.DataFrame(
            {
                "Actual UCS (kPa)": y_actual,
                "Predicted UCS (kPa)": y_pred,
                "Residual (kPa)": residuals,
            }
        )

        st.subheader("Prediction Results")
        st.dataframe(prediction_table, use_container_width=True)

        csv = prediction_table.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Prediction Results",
            data=csv,
            file_name="ucs_ann_predictions.csv",
            mime="text/csv",
        )


# ============================================================
# TAB 4: EXPLAINABLE AI
# ============================================================
with tab4:
    st.subheader("Explainable AI (SHAP)")

    if "trained_results" not in st.session_state:
        st.info("Please train the ANN model first.")
    else:
        results = st.session_state["trained_results"]
        final_model = results["final_model"]
        X_explain = results["X"].copy()

        st.write(
            "SHAP is used to estimate how each input feature contributes "
            "to the ANN model predictions."
        )

        max_samples = min(100, len(X_explain))
        explain_data = X_explain.sample(
            n=max_samples,
            random_state=int(random_state),
        )

        if st.button("🔍 Generate SHAP Analysis"):
            with st.spinner("Generating SHAP explanations..."):
                try:
                    # Model-agnostic SHAP explainer.
                    background_size = min(50, len(X_explain))
                    background = shap.sample(
                        X_explain,
                        background_size,
                        random_state=int(random_state),
                    )

                    explainer = shap.Explainer(
                        final_model.predict,
                        background,
                    )

                    shap_values = explainer(explain_data)

                    mean_abs_shap = np.abs(
                        shap_values.values
                    ).mean(axis=0)

                    importance_df = pd.DataFrame(
                        {
                            "Feature": selected_features,
                            "Mean |SHAP Value|": mean_abs_shap,
                        }
                    ).sort_values(
                        "Mean |SHAP Value|",
                        ascending=False,
                    )

                    st.session_state["shap_values"] = shap_values
                    st.session_state["shap_data"] = explain_data
                    st.session_state["importance_df"] = importance_df

                except Exception as e:
                    st.error(
                        "SHAP analysis could not be completed. "
                        f"Details: {e}"
                    )

        if "importance_df" in st.session_state:
            importance_df = st.session_state["importance_df"]

            st.subheader("Global Feature Importance")
            st.dataframe(importance_df, use_container_width=True)

            fig3, ax3 = plt.subplots(figsize=(8, 5))

            ordered = importance_df.sort_values(
                "Mean |SHAP Value|",
                ascending=True,
            )

            ax3.barh(
                ordered["Feature"],
                ordered["Mean |SHAP Value|"],
            )

            ax3.set_xlabel("Mean Absolute SHAP Value")
            ax3.set_ylabel("Input Feature")
            ax3.set_title("Global Feature Importance for UCS Prediction")

            st.pyplot(fig3)

            st.subheader("SHAP Summary Plot")

            try:
                shap_values = st.session_state["shap_values"]
                shap_data = st.session_state["shap_data"]

                fig4 = plt.figure(figsize=(9, 6))

                shap.summary_plot(
                    shap_values.values,
                    shap_data,
                    feature_names=selected_features,
                    show=False,
                )

                st.pyplot(fig4, bbox_inches="tight")

            except Exception as e:
                st.warning(f"SHAP summary plot could not be displayed: {e}")

        st.warning(
            "Research note: PI = LL − PL. Because LL, PL and PI are "
            "mathematically related, their separate SHAP importance should "
            "be interpreted carefully."
        )


# ============================================================
# TAB 5: UCS PREDICTION
# ============================================================
with tab5:
    st.subheader("Predict UCS for a New Soil Sample")

    if "trained_results" not in st.session_state:
        st.info("Please train the ANN model first.")
    else:
        results = st.session_state["trained_results"]
        final_model = results["final_model"]

        st.write(
            "Enter the soil index properties below. The ANN model will "
            "predict the UCS."
        )

        input_values = {}

        input_columns = st.columns(2)

        for i, feature in enumerate(selected_features):
            median_value = float(X[feature].median())

            with input_columns[i % 2]:
                input_values[feature] = st.number_input(
                    feature,
                    value=median_value,
                    format="%.4f",
                    key=f"prediction_{feature}",
                )

        if st.button("🔮 Predict UCS", type="primary"):
            new_data = pd.DataFrame(
                [input_values],
                columns=selected_features,
            )

            predicted_ucs = float(final_model.predict(new_data)[0])

            st.session_state["last_prediction"] = {
                "inputs": input_values,
                "predicted_ucs": predicted_ucs,
            }

            st.success(
                f"### Predicted UCS = {predicted_ucs:.2f} kPa"
            )

            st.dataframe(new_data, use_container_width=True)


# ============================================================
# TAB 6: GEMINI INTERPRETATION
# ============================================================
with tab6:
    st.subheader("Gemini AI Engineering Interpretation")

    st.write(
        "Gemini provides a language-based interpretation of the ANN "
        "prediction. It does not replace the trained ANN model."
    )

    if "last_prediction" not in st.session_state:
        st.info(
            "First predict UCS in the 'Predict UCS' tab, then return here "
            "for AI interpretation."
        )
    else:
        prediction_info = st.session_state["last_prediction"]

        st.metric(
            "ANN Predicted UCS",
            f"{prediction_info['predicted_ucs']:.2f} kPa",
        )

        if st.button("🤖 Generate Gemini Interpretation"):
            api_key = get_gemini_key()

            with st.spinner("Gemini is preparing the engineering interpretation..."):
                try:
                    interpretation = gemini_interpretation(
                        api_key=api_key,
                        features_dict=prediction_info["inputs"],
                        predicted_ucs=prediction_info["predicted_ucs"],
                        feature_names=selected_features,
                    )

                    st.session_state["gemini_interpretation"] = interpretation

                except Exception as e:
                    st.error(
                        "Gemini interpretation could not be generated. "
                        f"Details: {e}"
                    )

        if "gemini_interpretation" in st.session_state:
            st.markdown(
                st.session_state["gemini_interpretation"]
            )


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Research Application: UCS Prediction of Clayey Soils Using "
    "Index Properties, ANN, 5-Fold Cross-Validation and Explainable AI."
)
