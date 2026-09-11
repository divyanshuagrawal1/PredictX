import streamlit as st
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="PredictX",
    page_icon="🤖",
    layout="wide"
)
st.markdown("""
<style>

.main {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.block-container {
    max-width: 1200px;
    margin: auto;
    padding: 3rem 4rem;
}

h1 {
    text-align: center;
    font-size: 3rem !important;
    font-weight: 800 !important;
    margin-bottom: 0.2rem !important;
}

[data-testid="stFormSubmitButton"] {
    width: 100% !important;
}

[data-testid="stFormSubmitButton"] > div {
    width: 100% !important;
}

[data-testid="stFormSubmitButton"] button {
    width: 100% !important;
    min-width: 100% !important;
    color: #ffffff !important;
    background: #2563eb !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 1rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    transition: all 0.2s ease;
}

[data-testid="stFormSubmitButton"] button:hover {
    background: #1d4ed8 !important;
    transform: translateY(-2px);
}
.predictx-title {
    text-align: center;
    font-size: 4rem !important;
    font-weight: 800 !important;
    letter-spacing: -2px;
    margin-bottom: 0.2rem !important;
}

.predictx-title span {
    font-weight: 900;
}
.hero-subtitle {
    text-align: center;
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
}

.hero-subtitle span {
    font-size: 0.95rem;
    font-weight: 400;
}
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 1rem;
    margin-bottom: 0.8rem;
}
.stApp {
    background: linear-gradient(
        135deg,
        #0b1020 0%,
        #111827 50%,
        #0f172a 100%
    );
}
.stApp,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp p,
.stApp label {
    color: #f8fafc;
}

[data-testid="stCaptionContainer"] {
    color: #94a3b8 !important;
}

</style>
""", unsafe_allow_html=True)
# Load saved model
model = joblib.load("models/xgboost_model.pkl")

# Load saved preprocessor
preprocessor = joblib.load("models/preprocessor.pkl")

# Load saved threshold
with open("models/config.json", "r") as file:
    config = json.load(file)

threshold = config["threshold"]


# App title
st.markdown(
    """
    <h1 class="predictx-title">
        🤖 Predict<span>X</span>
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div class="hero-subtitle">
        AI-Powered Predictive Maintenance
        <br>
        <span>Detect machine failures before they happen</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.write(
    "Enter machine parameters to predict the probability of machine failure."
)

st.markdown(
    '<div class="section-title">⚙️ Machine Parameters</div>',
    unsafe_allow_html=True
)
# Machine inputs
with st.form("machine_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"]
        )

    with col1:
        process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=305.0,
        max_value=314.0,
        value=310.0
    )

    with col2:
        air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=295.0,
        max_value=305.0,
        value=300.0
    )

    with col2:
        rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=1000,
        max_value=3000,
        value=1500
    )

    with col1:
        torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=80.0,
        value=40.0,
        step=0.1
    )

    with col2:
        tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=253,
        value=100,
        step=1
    )


# Prediction button
    predict_button = st.form_submit_button(
    "🔍  Predict Machine Failure"
)
if predict_button:
    st.divider()
    st.subheader("Prediction Result")
    st.info(
    f"Prediction threshold: {threshold * 100:.0f}% — "
    f"Failure is predicted when the probability is at or above this level."
)

    input_data = pd.DataFrame({
        "Type": [machine_type],
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear]
    })

    # Preprocess input
    input_processed = preprocessor.transform(input_data)

    # Get failure probability
    failure_probability = model.predict_proba(input_processed)[0][1]

    # Apply threshold
    prediction = int(failure_probability >= threshold)

    # Display result
    if prediction == 1:
        st.error("⚠️ Machine Failure Predicted")
    else:
        st.success("✅ Machine is Normal")

    if failure_probability >= 0.70:
        st.error("🔴 Risk Level: High")
    elif failure_probability >= 0.30:
        st.warning("🟡 Risk Level: Medium")
    else:
        st.success("🟢 Risk Level: Low")

    if failure_probability >= 0.70:
        st.info("Recommended Action: Inspect the machine immediately.")
    elif failure_probability >= 0.30:
        st.info("Recommended Action: Schedule a maintenance inspection.")
    else:
        st.info("Recommended Action: Continue normal operation.")

    # Display probability
    chart_col1, chart_col2 = st.columns([1, 2])
    with chart_col1:
        st.metric(
        "Failure Probability",
        f"{failure_probability * 100:.2f}%"
    )

        normal_probability = 1 - failure_probability

        fig, ax = plt.subplots(figsize=(3, 3))

        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        ax.pie(
        [normal_probability, failure_probability],
        labels=["Normal", "Failure"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": "white", "fontweight": "bold"}
    )

        ax.set_title(
    "Machine Risk Distribution",
    color="white",
    fontweight="bold",
    pad=15
)

        st.pyplot(fig)
        plt.close(fig)

    with chart_col2:
        normal_data = pd.read_csv("data/raw/ai4i2020.csv")
        normal_data = normal_data[normal_data["Machine failure"] == 0]

        feature_names = [
        "Air Temperature",
        "Process Temperature",
        "Rotational Speed",
        "Torque",
        "Tool Wear"
    ]

        feature_values = [
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear
    ]

        normal_min = [
        normal_data["Air temperature [K]"].min(),
        normal_data["Process temperature [K]"].min(),
        normal_data["Rotational speed [rpm]"].min(),
        normal_data["Torque [Nm]"].min(),
        normal_data["Tool wear [min]"].min()
    ]

        normal_max = [
        normal_data["Air temperature [K]"].max(),
        normal_data["Process temperature [K]"].max(),
        normal_data["Rotational speed [rpm]"].max(),
        normal_data["Torque [Nm]"].max(),
        normal_data["Tool wear [min]"].max()
    ]

        normalized_values = [
        ((value - min_val) / (max_val - min_val)) * 100
        for value, min_val, max_val
        in zip(feature_values, normal_min, normal_max)
    ]

        fig, ax = plt.subplots(figsize=(6, 4))

        x = range(len(feature_names))

        ax.axhspan(0, 100, alpha=0.12, label="Normal Range")
        ax.axhspan(100, 120, alpha=0.10, label="Above Normal")
        ax.axhspan(-20, 0, alpha=0.10, label="Below Normal")

        ax.plot(
        x,
        normalized_values,
        marker="o",
        linewidth=2
    )

        for i, (value, percentage) in enumerate(
        zip(feature_values, normalized_values)
    ):
            ax.annotate(
            f"{value:g}",
            (i, percentage),
            xytext=(0, -18 if percentage > 85 else 10),
            textcoords="offset points",
            ha="center",
            fontweight="bold"
        )

        ax.axhline(100, linestyle="--", linewidth=1)

        x_labels = [
        f"Air Temperature\n{normal_min[0]:g} - {normal_max[0]:g} K",
        f"Process Temperature\n{normal_min[1]:g} - {normal_max[1]:g} K",
        f"Rotational Speed\n{normal_min[2]:g} - {normal_max[2]:g} rpm",
        f"Torque\n{normal_min[3]:g} - {normal_max[3]:g} Nm",
        f"Tool Wear\n{normal_min[4]:g} - {normal_max[4]:g} min"
    ]

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=0, fontsize=8)

        ax.set_ylabel("Position Relative to Normal Range (%)")
        ax.set_title("Current Values vs Historical Normal Range")

        ax.set_ylim(
        min(-20, min(normalized_values) - 10),
        max(120, max(normalized_values) + 10)
    )

        ax.legend(loc="lower right")

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)
        feature_status = []

        for name, value, min_val, max_val in zip(
    feature_names,
    feature_values,
    normal_min,
    normal_max
):
            range_width = max_val - min_val

            lower_limit = min_val + (0.20 * range_width)
            upper_limit = max_val - (0.20 * range_width)

            if value < min_val or value > max_val:
                status = "🔴 Outside Range"
            elif value <= lower_limit or value >= upper_limit:
                status = "🟡 Near Limit"
            else:
                status = "🟢 Normal"

            feature_status.append({
        "Feature": name,
        "Current": value,
        "Normal Min": min_val,
        "Normal Max": max_val,
        "Status": status
    })
    st.markdown(
    '<div class="section-title">📊 Feature Status</div>',
    unsafe_allow_html=True
)
    st.dataframe(
    pd.DataFrame(feature_status),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Feature": st.column_config.TextColumn(
            "Feature",
            width="medium"
        ),
        "Current": st.column_config.NumberColumn(
            "Current",
            format="%.0f"
        ),
        "Normal Min": st.column_config.NumberColumn(
            "Normal Min",
            format="%.0f"
        ),
        "Normal Max": st.column_config.NumberColumn(
            "Normal Max",
            format="%.0f"
        ),
        "Status": st.column_config.TextColumn(
            "Status",
            width="medium"
        )
    }
)




st.subheader("How PredictX Works")

st.write(
    "PredictX analyzes five machine operating parameters and machine type "
    "using a trained XGBoost model."
)

st.write(
    "The model estimates the probability of machine failure. "
    "A 30% decision threshold is used to classify the machine as "
    "Normal or Failure Predicted."
)
st.divider()
st.caption("PredictX | AI-Based Predictive Maintenance")
st.info("Model: XGBoost | Prediction Threshold: 30%")
st.subheader("Model Explainability")
st.write(
    "Feature importance shows which input variables contributed most "
    "to the model's predictions. Higher values indicate greater "
    "importance within the trained XGBoost model."
)
st.image(
    "models/feature_importance.png",
    caption="XGBoost Feature Importance"
)
st.write(
    "Torque is the most influential feature in the trained XGBoost model, "
    "followed by rotational speed and tool wear."
)
st.divider()
st.subheader("Model Performance")
performance_col1, performance_col2, performance_col3, performance_col4, performance_col5 = st.columns(5)

with performance_col1:
    st.metric("Accuracy", "98.65%")

with performance_col2:
    st.metric("Precision", "82%")

with performance_col3:
    st.metric("Recall", "77.94%")

with performance_col4:
    st.metric("F1-Score", "0.80")

with performance_col5:
    st.metric("ROC-AUC", "0.9765")
performance_labels = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

performance_values = [
    98.65,
    82.00,
    77.94,
    80.00,
    97.65
]

angles = np.linspace(
    0,
    2 * np.pi,
    len(performance_labels),
    endpoint=False
)

performance_values += performance_values[:1]
angles = np.concatenate((angles, [angles[0]]))

fig, ax = plt.subplots(
    figsize=(4, 4),
    subplot_kw=dict(polar=True)
)

ax.plot(
    angles,
    performance_values,
    linewidth=2
)

ax.fill(
    angles,
    performance_values,
    alpha=0.15
)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(performance_labels)

ax.set_ylim(0, 100)

ax.set_title(
    "XGBoost Model Performance",
    fontweight="bold",
    pad=20
)

st.pyplot(fig)
plt.close(fig)