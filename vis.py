import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import streamlit as st

def summarize_data(df):
    """
    Summarize the dataset:
    - Total number of data points
    - Number of data points for each college
    - Basic statistics for Raw College GPA (X)

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the data.

    Returns:
        None: Displays the summarized information using Streamlit.
    """
    # Total number of data points
    total_points = df.shape[0]
    
    # Number of data points per college
    points_per_college = df['College'].value_counts()
    
    # Basic statistics for Raw College GPA (X)
    raw_college_gpa_stats = df['Raw College GPA (X)'].describe()[['min', 'mean', 'max', 'std']]

    # Display results using Streamlit
    st.write(f"**Total number of data points:** {total_points}")

    st.write(points_per_college)

    st.write("Basic Statistics")
    st.dataframe(round(df.describe(),2))


# Display R^2 values
def display_r2(df):
    """Calculate and display the absolute R^2 values."""
    if 'University Performance GPA (Z)' in df.columns:
        r2_xz = abs(r2_score(df['Raw College GPA (X)'], df['University Performance GPA (Z)']))
        r2_yz = abs(r2_score(df['Moderated GPA (Y)'], df['University Performance GPA (Z)']))
    

# Plot histograms
def plot_histograms(df, column, hue, palette,bins=20, kde=True, alpha=0.5, ax=None, title=None):
    """Plot histograms with optional KDE and annotations for mean/std."""
    sns.histplot(data=df, x=column, hue=hue, bins=bins, kde=kde, alpha=alpha, element="step", ax=ax,legend=True,palette=palette)
    sns.move_legend(
    ax, "lower center",
    bbox_to_anchor=(0.5, 1.1), ncol=3, title=None, frameon=False,
)
    
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(title if title else column)

    # Add mean and std annotations
    stats_text = []
    for college, group_data in df.groupby(hue):
        mean = group_data[column].mean()
        std = group_data[column].std()
        ax.axvline(mean, linestyle="--", linewidth=1, label=f"{college} Mean",color='black')
        stats_text.append(f"{college}: \n-Mean={mean:.2f}, \n-Std={std:.2f}")
    
    ax.text(0.02, 0.98, "\n".join(stats_text), transform=ax.transAxes, fontsize=8.5, va="top", bbox=dict(facecolor="white", alpha=0.7))

def plot_scatter(ax, x, y, hue, title, xlabel, ylabel, r2):
    sns.scatterplot(x=x, y=y, hue=hue, ax=ax, alpha=0.7)
    ax.set_title(f"{title}\nR² = {abs(r2):.2f}", fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
  

def plot_sba_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), dpi=400)

    # Raw College GPA vs University Performance GPA
    r2_raw = r2_score(df['Raw College GPA (X)'], df['University Performance GPA (Z)'])
    plot_scatter(
        axes[0],
        df['University Performance GPA (Z)'],
        df['Raw College GPA (X)'],
        df['College'],
        "(a). X-Z Plot",
        "University Performance GPA (Z)",
        "Raw College GPA (X)",
        r2_raw
    )

    # Moderated GPA vs University Performance GPA
    r2_mod = r2_score(df['Moderated GPA (Y)'], df['University Performance GPA (Z)'])
    plot_scatter(
        axes[1],
        df['University Performance GPA (Z)'],
        df['Moderated GPA (Y)'],
        df['College'],
        "(b). Y-Z Plot",
        "University Performance GPA (Z)",
        "Moderated GPA (Y)",
        r2_mod
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return fig
