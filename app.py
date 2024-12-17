import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from models import train_hkeaa_model, predict_hkeaa_model, train_mean_adjust_model, predict_mean_adjust_model,train_hkeaa_model_1,predict_hkeaa_model_1
from vis import display_r2, plot_histograms, summarize_data, plot_sba_analysis


# Streamlit UI
st.set_page_config(layout="wide")
st.title("Standardizing College GPA for University Admissions: A Moderation Tool for Fairer Intake Decisions")

with st.container():
    st.markdown(r"""
    For this tool to work, ensure that your datasets meet the following requirements:

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
st.subheader("Step 0: Select Moderation Model")
model_options = {
    "Inter-Score Regression with Pooled Standardization": (train_hkeaa_model, predict_hkeaa_model),
    "Inter-Score Regression with Benchmark Standardization": (train_hkeaa_model_1, predict_hkeaa_model_1),
    "Mean Adjust Model": (train_mean_adjust_model, predict_mean_adjust_model)
}
selected_model = st.selectbox("Choose a model for moderation:", list(model_options.keys()))

train_model_func, predict_model_func = model_options[selected_model]

MODEL_STATS = None
col1,col2=st.columns(2)
with col1:
    # Step 1: Upload training data
    st.subheader("Step 1: Upload Training Data")
    training_file = st.file_uploader("Upload a CSV/Excel file with 'College', 'Raw College GPA (X)', 'University Performance GPA (Z)'", type=["csv", "xlsx"])


    if training_file:
        df_train = pd.read_csv(training_file) if training_file.name.endswith(".csv") else pd.read_excel(training_file)
        summarize_data(df_train)
        if {'College', 'Raw College GPA (X)', 'University Performance GPA (Z)'}.issubset(df_train.columns):
            st.success("Training file loaded successfully!")
            MODEL_STATS = train_model_func(df_train)

            with col2:
                st.subheader('Training Dataset Results')
                # Display model parameters
                st.write(MODEL_STATS)
                # Display histograms
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
                display_r2(df_train)
                r2_fig=plot_sba_analysis(df_train)
                r2_fig.suptitle(f"{selected_model}")

                st.pyplot(r2_fig)
                
        else:
            st.error("File must contain 'College', 'Raw College GPA (X)', and 'University Performance GPA (Z)'.")
with col1:
    # Step 2: Upload test data
    st.subheader("Step 2: Upload Test Data")
    test_file = st.file_uploader("Upload a CSV/Excel file with 'College' and 'Raw College GPA (X)'", type=["csv", "xlsx"])

    if test_file and MODEL_STATS:
        df_test = pd.read_csv(test_file) if test_file.name.endswith(".csv") else pd.read_excel(test_file)
        summarize_data(df_test)
        if {'College', 'Raw College GPA (X)'}.issubset(df_test.columns):
            st.success("Test file loaded successfully!")
            df_result = predict_model_func(df_test, MODEL_STATS)

            with col2:
                st.subheader('Testing Dataset Results')
                if {'University Performance GPA (Z)'}.issubset(df_train.columns):
                    display_r2(df_result)
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