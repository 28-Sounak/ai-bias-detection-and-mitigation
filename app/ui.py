import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from core.bias_detection import demographic_parity, disparate_impact
from core.mitigation import balance_data
from core.model import train_model
from core.gemini_explainer import explain_bias


# ---------------- PLOTTING FUNCTION ----------------
def plot_rates(rates, title):
    groups = list(rates.keys())
    values = list(rates.values())

    fig, ax = plt.subplots()

    ax.bar(groups, values)
    ax.set_title(title)
    ax.set_ylabel("Positive Outcome Rate")

    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')

    st.pyplot(fig)


# ---------------- MAIN UI ----------------
def render_ui():
    st.title("AI FAIRNESS AUDIT TOOL")
    st.write("Detect and Mitigate Bias in Datasets")

    file = st.file_uploader("Upload your CSV Dataset", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        columns = df.columns.tolist()

        target = st.selectbox("Select Target Column (0/1)", columns)
        sensitive = st.selectbox("Select Sensitive Feature", columns)

        # Ensure numeric target
        df[target] = pd.to_numeric(df[target], errors='coerce')

        if st.button("Analyze Bias"):

            # ---------------- BEFORE MITIGATION ----------------
            rates = demographic_parity(df, target, sensitive)
            di = disparate_impact(rates)

            st.subheader("Before Mitigation")
            st.write("Group Rates:", rates)
            st.metric("Disparate Impact", f"{di:.2f}")

            plot_rates(rates, "Before Mitigation")

            if di < 0.8:
                st.error("BIAS DETECTED!")
            else:
                st.success("NO SIGNIFICANT BIAS DETECTED")

            # ---------------- MITIGATION ----------------
            df_balanced = balance_data(df, target, sensitive)

            st.write("Columns after balancing:", list(df_balanced.columns))

            new_rates = demographic_parity(df_balanced, target, sensitive)
            new_di = disparate_impact(new_rates)

            st.subheader("After Mitigation")
            st.write("New Group Rates:", new_rates)
            st.metric("New Disparate Impact", f"{new_di:.2f}")

            plot_rates(new_rates, "After Mitigation")

            if new_di > di:
                st.success("Bias Reduced!")
            else:
                st.warning("No Improvement Detected")

            # ---------------- MODEL ANALYSIS ----------------
            st.subheader("Model Bias Analysis")

            model_df = train_model(df, target, sensitive)

            model_rates = demographic_parity(model_df, "prediction", sensitive)
            model_di = disparate_impact(model_rates)

            st.write("Model Prediction Rates:", model_rates)
            st.metric("Model Disparate Impact", f"{model_di:.2f}")

            plot_rates(model_rates, "Model Bias")

            if model_di < 0.8:
                st.error("Model is Biased")
            else:
                st.success("Model is Fair")

            # ---------------- GEMINI EXPLANATION ----------------
            st.subheader("AI Explanation (Gemini)")

            try:
                explanation = explain_bias(model_rates, model_di)
                st.write(explanation)
            except Exception as e:
                st.warning("Gemini explanation failed. Check API key or internet connection.")
                #st.error(f"Gemini Error: {e}")
                if model_di < 0.8:
                    st.write(
                        "The model shows bias because one group receives significantly more favorable outcomes than others. "
                        "This indicates unfair decision-making patterns."
                    )
                else:
                    st.write(
                        "The model appears fair as the outcomes are evenly distributed across different groups."
                    )