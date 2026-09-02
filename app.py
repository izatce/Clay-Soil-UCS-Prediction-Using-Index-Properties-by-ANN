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
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Soil UCS Prediction Using ANN",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Soil UCS Prediction Using ANN")
st.caption(
    "Automatic ANN Architecture Optimization, 5-Fold Cross-Validation "
    "and Explainable AI for UCS Prediction"
)

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

# The application automatically compares these architectures.
# Users cannot manually change hidden-layer neurons.
CANDIDATE_ARCHITECTURES = [
    (8,),
    (16,),
    (32,),
    (8, 4),
    (16, 8),
    (32, 16),
    (64, 32),
]


def clean_columns(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_excel(file):
    return clean_columns(pd.read_excel(file))


def calculate_metrics(y_true, y_pred):
    return {
        "R²": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def build_ann(architecture, max_iter=2000, random_state=42):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "ann",
                MLPRegressor(
                    hidden_layer_sizes=architecture,
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


def evaluate_architecture(X, y, architecture, max_iter, random_state):
    """Evaluate one ANN architecture using 5-fold cross-validation."""
    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=int(random_state),
    )

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
        model = build_ann(
            architecture=architecture,
            max_iter=max_iter,
            random_state=int(random_state) + fold,
        )

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        fold_metrics.append(calculate_metrics(y_test, predictions))

    results = pd.DataFrame(fold_metrics)

    return {
        "Architecture": str(architecture),
        "Hidden Layers": len(architecture),
        "Total Neurons": sum(architecture),
        "Mean R²": results["R²"].mean(),
        "Std R²": results["R²"].std(),
        "Mean MAE": results["MAE"].mean(),
        "Mean RMSE": results["RMSE"].mean(),
        "Mean MAPE (%)": results["MAPE (%)"].mean(),
    }


def get_gemini_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


def get_gemini_interpretation(api_key, inputs, predicted_ucs, features):
    if not api_key:
        return (
            "Gemini API key was not found. Add GEMINI_API_KEY "
            "to Streamlit Secrets."
        )

    prompt = f"""
You are assisting with geotechnical engineering research.

An Artificial Neural Network predicted the Unconfined Compressive
Strength (UCS) of a clayey soil.

Input properties:
{inputs}

ANN predicted UCS:
{predicted_ucs:.2f} kPa

Input features:
{", ".join(features)}

Provide a concise academic engineering interpretation.

Important:
- The ANN, not Gemini, produced the numerical prediction.
- Discuss the result cautiously.
- Explain the possible relevance of plasticity, clay fraction,
  clay activity, water content and specific gravity.
- Do not make unsupported causal claims.
- Mention that PI = LL - PL, so these related features should be
  interpreted carefully.
"""

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ ANN Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Soil Dataset (Excel)",
    type=["xlsx", "xls"],
)

# IMPORTANT: Hidden-layer inputs are intentionally NOT provided.
st.sidebar.success("🤖 Automatic ANN Optimization Enabled")

st.sidebar.write(
    "The application automatically evaluates multiple ANN hidden-layer "
    "architectures using 5-fold cross-validation."
)

st.sidebar.write("**Candidate architectures:**")
st.sidebar.code(
    "\n".join(str(a) for a in CANDIDATE_ARCHITECTURES)
)

max_iter = st.sidebar.number_input(
    "Maximum Training Iterations",
    min_value=500,
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

st.sidebar.info(
    "Hidden-layer sizes are selected automatically. "
    "Manual modification is disabled."
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "🔐 Store GEMINI_API_KEY in Streamlit Secrets. "
    "Do not upload your API key to GitHub."
)


# ============================================================
# LOAD DATA
# ============================================================
if uploaded_file is None:
    st.info("👈 Upload your Excel soil dataset to begin.")

    st.write("### Default Input Features")
    st.write(DEFAULT_FEATURES)

    st.write("### Target")
    st.write(TARGET_NAME)

    st.stop()

try:
    data = load_excel(uploaded_file)
except Exception as e:
    st.error(f"Unable to read the Excel file: {e}")
    st.stop()


tabs = st.tabs(
    [
        "📁 Dataset",
        "🧠 Automatic ANN Optimization",
        "📊 Performance",
        "🔍 Explainable AI",
        "🔮 Predict UCS",
        "🤖 Gemini Interpretation",
    ]
)

# ============================================================
# DATASET TAB
# ============================================================
with tabs[0]:
    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", data.shape[0])
    c2.metric("Columns", data.shape[1])
    c3.metric("Missing Values", int(data.isna().sum().sum()))

    st.dataframe(data.head(10), use_container_width=True)

    all_columns = list(data.columns)
    default_features = [
        f for f in DEFAULT_FEATURES if f in all_columns
    ]

    selected_features = st.multiselect(
        "Select Input Features",
        all_columns,
        default=default_features,
    )

    target_index = (
        all_columns.index(TARGET_NAME)
        if TARGET_NAME in all_columns
        else len(all_columns) - 1
    )

    target = st.selectbox(
        "Select Target Output",
        all_columns,
        index=target_index,
    )

    st.dataframe(data.describe(include="all").T, use_container_width=True)

# Keep selections available outside the tab.
selected_features = st.session_state.get(
    "Select Input Features",
    [f for f in DEFAULT_FEATURES if f in data.columns],
)

# Streamlit auto-generates keys from labels; retrieve values directly
# by ensuring fallback values when no explicit key is present.
if not selected_features:
    selected_features = [
        f for f in DEFAULT_FEATURES if f in data.columns
    ]

target = st.session_state.get(
    "Select Target Output",
    TARGET_NAME if TARGET_NAME in data.columns else data.columns[-1],
)

# Use widget values stored through explicit session-safe retrieval.
# The labels above are sufficient for Streamlit state.
if target not in data.columns:
    target = TARGET_NAME if TARGET_NAME in data.columns else data.columns[-1]

if len(selected_features) < 2 or target in selected_features:
    st.error(
        "Select at least two input features and ensure the target is not an input."
    )
    st.stop()

model_data = data[selected_features + [target]].copy()

for col in selected_features + [target]:
    model_data[col] = pd.to_numeric(model_data[col], errors="coerce")

model_data = model_data.dropna(subset=[target])

X = model_data[selected_features].reset_index(drop=True)
y = model_data[target].reset_index(drop=True)


# ============================================================
# AUTOMATIC OPTIMIZATION TAB
# ============================================================
with tabs[1]:
    st.subheader("Automatic ANN Architecture Optimization")

    st.markdown(
        """
The hidden-layer architecture is **fully automatic**.

You do not enter the number of neurons manually. The application:

1. Tests predefined ANN architectures.
2. Applies **5-fold cross-validation** to each architecture.
3. Calculates R², MAE, RMSE and MAPE.
4. Ranks the models primarily by **lowest Mean RMSE**.
5. Uses R² and model simplicity as additional selection criteria.
6. Automatically selects the optimal architecture.
"""
    )

    if st.button(
        "🔍 Automatically Find Optimal ANN Architecture",
        type="primary",
    ):
        rows = []
        progress = st.progress(0)
        status = st.empty()

        for i, architecture in enumerate(CANDIDATE_ARCHITECTURES):
            status.write(
                f"Testing ANN architecture: {architecture}"
            )

            row = evaluate_architecture(
                X=X,
                y=y,
                architecture=architecture,
                max_iter=int(max_iter),
                random_state=int(random_state),
            )
            rows.append(row)

            progress.progress(
                int((i + 1) / len(CANDIDATE_ARCHITECTURES) * 100)
            )

        progress.empty()
        status.empty()

        results_df = pd.DataFrame(rows)

        # Lowest RMSE is the main criterion.
        # Higher R² and fewer neurons break close ties.
        results_df = results_df.sort_values(
            by=["Mean RMSE", "Mean R²", "Total Neurons"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

        results_df.insert(
            0,
            "Rank",
            range(1, len(results_df) + 1),
        )

        architecture_map = {
            str(a): a for a in CANDIDATE_ARCHITECTURES
        }

        best_text = results_df.loc[0, "Architecture"]
        best_architecture = architecture_map[best_text]

        st.session_state["optimization_results"] = results_df
        st.session_state["best_architecture"] = best_architecture

        # Clear old model results if architecture is recalculated.
        st.session_state.pop("trained_results", None)
        st.session_state.pop("shap_values", None)
        st.session_state.pop("importance_df", None)

        st.success(
            f"🏆 Optimal ANN Architecture: {best_architecture}"
        )

    if "optimization_results" in st.session_state:
        comparison = st.session_state["optimization_results"]

        st.subheader("Architecture Comparison")

        st.dataframe(
            comparison.style.format(
                {
                    "Mean R²": "{:.4f}",
                    "Std R²": "{:.4f}",
                    "Mean MAE": "{:.4f}",
                    "Mean RMSE": "{:.4f}",
                    "Mean MAPE (%)": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        best = comparison.iloc[0]

        st.subheader("🏆 Automatically Selected Model")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Architecture",
            str(st.session_state["best_architecture"]),
        )
        c2.metric("Mean R²", f"{best['Mean R²']:.4f}")
        c3.metric("Mean RMSE", f"{best['Mean RMSE']:.4f}")
        c4.metric("Mean MAE", f"{best['Mean MAE']:.4f}")

    st.markdown("---")
    st.subheader("Train Final ANN Model")

    if "best_architecture" not in st.session_state:
        st.warning(
            "First click 'Automatically Find Optimal ANN Architecture'."
        )
    else:
        st.info(
            "The final model will use the automatically selected "
            f"architecture: {st.session_state['best_architecture']}"
        )

        if st.button("🚀 Train Final ANN Model", type="primary"):
            architecture = st.session_state["best_architecture"]

            kfold = KFold(
                n_splits=5,
                shuffle=True,
                random_state=int(random_state),
            )

            all_predictions = np.zeros(len(y))
            fold_rows = []

            with st.spinner("Training final ANN model..."):
                for fold, (train_idx, test_idx) in enumerate(
                    kfold.split(X), start=1
                ):
                    model = build_ann(
                        architecture,
                        int(max_iter),
                        int(random_state) + fold,
                    )

                    model.fit(
                        X.iloc[train_idx],
                        y.iloc[train_idx],
                    )

                    prediction = model.predict(
                        X.iloc[test_idx]
                    )

                    all_predictions[test_idx] = prediction

                    metrics = calculate_metrics(
                        y.iloc[test_idx],
                        prediction,
                    )
                    metrics["Fold"] = fold
                    fold_rows.append(metrics)

                final_model = build_ann(
                    architecture,
                    int(max_iter),
                    int(random_state),
                )

                final_model.fit(X, y)

            st.session_state["trained_results"] = {
                "final_model": final_model,
                "predictions": all_predictions,
                "actual": y.values,
                "fold_results": pd.DataFrame(fold_rows),
                "overall_metrics": calculate_metrics(
                    y,
                    all_predictions,
                ),
                "architecture": architecture,
                "features": selected_features,
                "X": X.copy(),
            }

            st.success("Final ANN model trained successfully.")

    if "trained_results" in st.session_state:
        folds = st.session_state["trained_results"]["fold_results"]

        st.subheader("Final 5-Fold Results")
        st.dataframe(folds, use_container_width=True)

        mean_metrics = folds[
            ["R²", "MAE", "RMSE", "MAPE (%)"]
        ].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean R²", f"{mean_metrics['R²']:.4f}")
        c2.metric("Mean MAE", f"{mean_metrics['MAE']:.4f}")
        c3.metric("Mean RMSE", f"{mean_metrics['RMSE']:.4f}")
        c4.metric("Mean MAPE", f"{mean_metrics['MAPE (%)']:.2f}%")


# ============================================================
# PERFORMANCE TAB
# ============================================================
with tabs[2]:
    st.subheader("Model Performance")

    if "trained_results" not in st.session_state:
        st.info("Train the final ANN model first.")
    else:
        result = st.session_state["trained_results"]
        metrics = result["overall_metrics"]

        st.write(
            f"**Automatically selected architecture:** "
            f"`{result['architecture']}`"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{metrics['R²']:.4f}")
        c2.metric("MAE", f"{metrics['MAE']:.4f}")
        c3.metric("RMSE", f"{metrics['RMSE']:.4f}")
        c4.metric("MAPE", f"{metrics['MAPE (%)']:.2f}%")

        actual = result["actual"]
        predicted = result["predictions"]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(actual, predicted, alpha=0.75)

        low = min(actual.min(), predicted.min())
        high = max(actual.max(), predicted.max())

        ax.plot([low, high], [low, high], linestyle="--")
        ax.set_xlabel("Actual UCS (kPa)")
        ax.set_ylabel("Predicted UCS (kPa)")
        ax.set_title("Actual vs Predicted UCS")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        residual = actual - predicted

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.scatter(predicted, residual, alpha=0.75)
        ax2.axhline(0, linestyle="--")
        ax2.set_xlabel("Predicted UCS (kPa)")
        ax2.set_ylabel("Residual (kPa)")
        ax2.set_title("Residual Plot")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)


# ============================================================
# SHAP TAB
# ============================================================
with tabs[3]:
    st.subheader("Explainable AI (SHAP)")

    if "trained_results" not in st.session_state:
        st.info("Train the final ANN model first.")
    else:
        result = st.session_state["trained_results"]
        final_model = result["final_model"]
        X_explain = result["X"]
        features = result["features"]

        if st.button("🔍 Generate SHAP Analysis"):
            with st.spinner("Generating SHAP explanations..."):
                try:
                    background = shap.sample(
                        X_explain,
                        min(50, len(X_explain)),
                        random_state=int(random_state),
                    )

                    explain_data = X_explain.sample(
                        min(100, len(X_explain)),
                        random_state=int(random_state),
                    )

                    explainer = shap.Explainer(
                        final_model.predict,
                        background,
                    )

                    shap_values = explainer(explain_data)

                    importance = pd.DataFrame(
                        {
                            "Feature": features,
                            "Mean |SHAP Value|": np.abs(
                                shap_values.values
                            ).mean(axis=0),
                        }
                    ).sort_values(
                        "Mean |SHAP Value|",
                        ascending=False,
                    )

                    st.session_state["shap_values"] = shap_values
                    st.session_state["shap_data"] = explain_data
                    st.session_state["importance_df"] = importance

                except Exception as e:
                    st.error(f"SHAP analysis error: {e}")

        if "importance_df" in st.session_state:
            importance = st.session_state["importance_df"]

            st.dataframe(importance, use_container_width=True)

            fig3, ax3 = plt.subplots(figsize=(8, 5))
            plot_data = importance.sort_values(
                "Mean |SHAP Value|"
            )

            ax3.barh(
                plot_data["Feature"],
                plot_data["Mean |SHAP Value|"],
            )

            ax3.set_xlabel("Mean Absolute SHAP Value")
            ax3.set_title("Global Feature Importance")
            st.pyplot(fig3)


# ============================================================
# PREDICTION TAB
# ============================================================
with tabs[4]:
    st.subheader("Predict UCS for a New Soil Sample")

    if "trained_results" not in st.session_state:
        st.info("Train the final ANN model first.")
    else:
        result = st.session_state["trained_results"]
        model = result["final_model"]
        features = result["features"]

        values = {}
        columns = st.columns(2)

        for i, feature in enumerate(features):
            with columns[i % 2]:
                values[feature] = st.number_input(
                    feature,
                    value=float(X[feature].median()),
                    format="%.4f",
                    key=f"new_{feature}",
                )

        if st.button("🔮 Predict UCS", type="primary"):
            new_sample = pd.DataFrame([values])

            prediction = float(
                model.predict(new_sample)[0]
            )

            st.session_state["last_prediction"] = {
                "inputs": values,
                "predicted_ucs": prediction,
            }

            st.success(
                f"### Predicted UCS = {prediction:.2f} kPa"
            )


# ============================================================
# GEMINI TAB
# ============================================================
with tabs[5]:
    st.subheader("Gemini AI Engineering Interpretation")

    if "last_prediction" not in st.session_state:
        st.info("First predict UCS in the Predict UCS tab.")
    else:
        info = st.session_state["last_prediction"]

        st.metric(
            "ANN Predicted UCS",
            f"{info['predicted_ucs']:.2f} kPa",
        )

        if st.button("🤖 Generate Gemini Interpretation"):
            api_key = get_gemini_key()

            try:
                text = get_gemini_interpretation(
                    api_key,
                    info["inputs"],
                    info["predicted_ucs"],
                    st.session_state["trained_results"]["features"],
                )

                st.session_state["gemini_text"] = text

            except Exception as e:
                st.error(
                    f"Gemini interpretation error: {e}"
                )

        if "gemini_text" in st.session_state:
            st.markdown(st.session_state["gemini_text"])


st.markdown("---")
st.caption(
    "Soil UCS Prediction | Automatic ANN Architecture Optimization | "
    "5-Fold Cross-Validation | SHAP Explainable AI"
)
