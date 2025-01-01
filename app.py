import numpy as np
import pandas as pd
import streamlit as st
from vis import display_r2, plot_histograms, summarize_data, plot_sba_analysis

import matplotlib.pyplot as plt
from models import (train_mean_adjust_model, predict_mean_adjust_model,
                   train_reg_std_model,predict_reg_std_model,
                   train_linear_regression_model, predict_linear_regression_model) 


# Streamlit UI
# Set the page layout with an expanded configuration
st.set_page_config(page_title="GPA Moderation Tool", layout="wide", page_icon="📊")

# Sidebar for navigation and instructions
st.sidebar.title("📊 GPA Moderation Tool")
st.sidebar.markdown("""
This interactive tool provides methods for moderating college GPA to enhance fairness and equity in university admissions. 
Follow the steps below to upload your datasets and analyze the results.
""")

st.sidebar.subheader("Instructions")
st.sidebar.markdown("""
1. **Select Moderation Model**: Choose a model for GPA moderation.
2. **Upload Training Data**: Upload a CSV/Excel file with the required columns.
3. **Upload Test Data**: Upload a CSV/Excel file with the required columns.
4. **Analyze Results**: View and download the moderated GPA results.
""")

# Main page layout
st.title("📊 Standardizing College GPA for University Admissions")
st.markdown("""
This tool helps in moderating college GPA to ensure fairness in university admissions. 
Please follow the steps below to upload your datasets and analyze the results.
""")


with st.expander("See Dataset Requirements for Training and Testing"):
    st.markdown(r"""
    ### Training Dataset:
    The training dataset must include the following columns:

    | Column Name                   | Description                              | Example       |
    |-------------------------------|------------------------------------------|---------------|
    | `College`                     | Name of the college (categorical)        | `College 1`, `College 2` |
    | `Raw College GPA (X)`         | Raw GPA scores for students (0 - 4.33)   | `2.5`, `3.1`, `4.0`      |
    | `University Performance GPA (Z)` | University performance GPA (0 - 4.33)   | `2.8`, `3.5`, `4.2`      |

    **Example Training Data:**
    | College   | Raw College GPA (X) | University Performance GPA (Z) |
    |-----------|----------------------|-------------------------------|
    | College 1 | 2.5                  | 2.8                           |
    | College 1 | 3.1                  | 3.5                           |
    | College 1 | 4.0                  | 4.2                           |
    | College 2 | 2.8                  | 3.0                           |
    | College 2 | 3.3                  | 3.7                           |
    """)

    st.markdown(r"""
    ### Test Dataset:
    The test dataset must include the following columns:

    | Column Name                    | Description                              | Example       |
    |--------------------------------|------------------------------------------|---------------|
    | `College`                      | Name of the college (categorical)        | `College 1`, `College 2` |
    | `Raw College GPA (X)`          | Raw GPA scores for students (0 - 4.33)   | `2.5`, `3.0`, `4.1`      |
    | *(Optional)* `University Performance GPA (Z)` | University GPA for verification (0 - 4.33) | `2.8`, `3.6`  |

    **Notes:**
    - If the `University Performance GPA (Z)` column is **not provided**, predictions can still be made for the `Moderated GPA (Y)`.
    - If `University Performance GPA (Z)` is provided in the test dataset, it can be used for **verification** purposes.
    """)

# Dropdown for model selection
with st.expander("GPA Moderation Methods"):
    st.markdown(r"""

    ### Method 1: Mean-Adjusted Model (Baseline)

    This is the simplest method that adjusts raw college GPAs by aligning them with the University Performance GPA mean. The moderated GPA is calculated as:

    $$Y = \bar{Z} + (X - \bar{X})$$

    Where:
    - $Y$: Moderated college GPA
    - $\bar{Z}$: Group mean of university GPA, $\bar{Z} = \text{mean}(Z|\text{College})$
    - $\bar{X}$: Group mean of raw college GPA, $\bar{X} = \text{mean}(X|\text{College})$
    - $X$: Raw college GPA

    ### Method 2: Direct Benchmark-to-Raw Score Regression

    This method uses linear regression to calibrate college GPAs against university performance GPAs. The moderated GPA is obtained by:

    $$Y = kX + b$$

    Where:
    - $k$: Slope coefficient capturing the linear relationship between $X$ and $Z$
    - $b$: Intercept parameter
    - $(k,b) = \arg\min_{(k,b)} \sum_i (Z_i - Y_i)^2$: Coefficients optimized by minimizing squared differences
    - $Z_i$: University GPA for student $i$
    - $X_i$: Raw college GPA for student $i$

    ### Method 3: Inter-Score Regression with Benchmark Standardization

    This method transforms raw college GPAs into moderated GPAs while incorporating university performance GPA standard deviation:

    $$Y = \bar{X} + \beta(\bar{Z} - \bar{Z}_\text{mean}) + (X - \bar{X})\frac{S_Z}{S_X}$$

    Where:
    - $\bar{X}$: Global mean of raw college GPA
    - $\beta$: Regression coefficient between raw college GPA and university GPA (0 to 1)
    - $\bar{Z}_\text{mean}$: Global mean of university GPA
    - $S_X$: Group standard deviation of raw college GPA, $S_X = \text{std}(X|\text{College})$
    - $S_Z$: Group standard deviation of university GPA, $S_Z = \text{std}(Z|\text{College})$""")

st.subheader("Step 1: Select Moderation Model")
model_options = {
    "3- Inter-Score Regression with Benchmark Standardization": (train_reg_std_model, predict_reg_std_model),
    '2- Inter-Score Regression': (train_linear_regression_model, predict_linear_regression_model),
    "1- Mean Adjust Model": (train_mean_adjust_model, predict_mean_adjust_model)
}

selected_model = st.selectbox("Choose a model for moderation:", list(model_options.keys()))
train_model_func, predict_model_func = model_options[selected_model]

MODEL_STATS = None

col1,col2=st.columns(2)

with col1:
    # Step 1: Upload training data
    st.divider()
    st.subheader("Step 1: Upload Training Data")
    training_file = st.file_uploader("Upload a CSV/Excel file with 'College', 'Raw College GPA (X)', 'University Performance GPA (Z)'", type=["csv", "xlsx"])

    if training_file:
        df_train = pd.read_csv(training_file) if training_file.name.endswith(".csv") else pd.read_excel(training_file)
        summarize_data(df_train)
        if {'College', 'Raw College GPA (X)', 'University Performance GPA (Z)'}.issubset(df_train.columns):
            st.success("Training file loaded successfully!")
            MODEL_STATS = train_model_func(df_train)
             # Apply the model to the training data
            df_train = predict_model_func(df_train, MODEL_STATS)

            with col2:
                st.divider()
                st.subheader('Training Dataset Results')
                st.write('Model Coefficients')
                # Display model parameters
                st.write(MODEL_STATS)
                # Display histograms
                st.write('**Distribution of Raw College GPA (X), Moderated GPA (Y) and University GPA (Z)**')
                fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=600)
                fig.suptitle(f"Results for: {selected_model}", fontsize=12)
                plot_histograms(df_train, "Raw College GPA (X)", "College", ax=axes[0], title="(a).Raw College GPA",palette='bright')
                plot_histograms(df_train, "University Performance GPA (Z)", "College", ax=axes[1], title="(b).University GPA",palette='dark')
                plot_histograms(df_train, "Moderated GPA (Y)", "College", ax=axes[2], title="(c).Moderated College GPA",palette='bright')
                # After all plots, unify the y-axis limits based on the highest upper limit found
                max_y = max(ax.get_ylim()[1] for ax in axes)  # get the largest top y-limit
                for ax in axes:
                    ax.set_ylim(0, max_y)
                plt.tight_layout()
                plt.show()
                st.pyplot(fig)
                r2_xz, r2_yz = display_r2(df_train)  # Calculate R² values
                delta_r = round(r2_yz - r2_xz,2)
                st.write(f"R² (XZ) = {r2_xz}, R² (YZ) = {r2_yz}, ΔR² = {delta_r}")
                r2_fig=plot_sba_analysis(df_train)
                r2_fig.suptitle(f"{selected_model}")
                st.markdown("**Coefficient of Determination (R2)**")

                st.pyplot(r2_fig)    
        else:
            st.error("File must contain 'College', 'Raw College GPA (X)', and 'University Performance GPA (Z)'.")
with col1:
    # Step 2: Upload test data
    st.divider()
    st.subheader("Step 2: Upload Test Data")
    test_file = st.file_uploader("Upload a CSV/Excel file with 'College' and 'Raw College GPA (X)'", type=["csv", "xlsx"])

    if test_file and MODEL_STATS:
        df_test = pd.read_csv(test_file) if test_file.name.endswith(".csv") else pd.read_excel(test_file)
        summarize_data(df_test)
        if {'College', 'Raw College GPA (X)'}.issubset(df_test.columns):
            st.success("Test file loaded successfully!")
            df_result = predict_model_func(df_test, MODEL_STATS)

            with col2:
                st.divider()
                st.subheader('Testing Dataset Results')
                if {'University Performance GPA (Z)'}.issubset(df_test.columns):
                    r2_xz, r2_yz = display_r2(df_test)  # Calculate R² values
                    delta_r = round(r2_yz - r2_xz,2)
                    st.write(f"R² (XZ) = {r2_xz}, R² (YZ) = {r2_yz}, ΔR² = {delta_r}")
                fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), dpi=400)
                plot_histograms(df_result, "Raw College GPA (X)", "College", ax=axes[0], title="(a).Raw College GPA",palette='bright')
                plot_histograms(df_result, "Moderated GPA (Y)", "College", ax=axes[1], title="(b).Moderated College GPA",palette='bright')
                # After all plots, unify the y-axis limits based on the highest upper limit found
                max_y = max(ax.get_ylim()[1] for ax in axes)  # get the largest top y-limit
                for ax in axes:
                    ax.set_ylim(0, max_y)

                plt.tight_layout()
                plt.show()
                st.pyplot(fig)
                st.write('Moderated Score')
                df_result['Moderated GPA (Y)'] = np.maximum(df_result['Moderated GPA (Y)'], 0)     
                df_result = df_result.apply(lambda x: round(x, 2) if pd.api.types.is_numeric_dtype(x) else x)
                st.write(df_result)

            # Download button
            st.download_button(
                label="Download Results as CSV",
                data=df_result.to_csv(index=False).encode("utf-8"),
                file_name="moderated_gpa_output.csv",
                mime="text/csv",
            )
        else:
            st.error("Test file must contain 'College' and 'Raw College GPA (X)'.")
