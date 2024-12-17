import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

def train_hkeaa_model(df):
    """Train the moderation model and return the calculated statistics."""
    MODEL_STATS = {}
    
    # Train regression model for beta
    model = LinearRegression()
    model.fit(df[['University Performance GPA (Z)']], df['Raw College GPA (X)'])
    beta = model.coef_[0]

    # Compute required statistics
    X_mean = df['Raw College GPA (X)'].mean()
    z_mean = df['University Performance GPA (Z)'].mean()
    z_bars = df.groupby('College')['University Performance GPA (Z)'].mean()
    x_bars = df.groupby('College')['Raw College GPA (X)'].mean()
    s_x = df.groupby('College')['Raw College GPA (X)'].std()
    s_z = df.groupby('College')['University Performance GPA (Z)'].std()
    s_p = np.sqrt(0.5 * s_x**2 + 0.5 * s_z**2)

    # Map group-level statistics to the DataFrame
    df['z_bar'] = df['College'].map(z_bars)
    df['x_bar'] = df['College'].map(x_bars)
    df['s_p'] = df['College'].map(s_p)
    df['s_x'] = df['College'].map(s_x)

    # Calculate moderated GPA scores
    df['Moderated GPA (Y)'] = (
        X_mean +
        beta * (df['z_bar'] - z_mean) +
        (df['Raw College GPA (X)'] - df['x_bar']) * (df['s_p'] / df['s_x'])
    )
    df['Moderated GPA (Y)'] = df['Moderated GPA (Y)'].clip(lower=0)

    # Store model parameters
    MODEL_STATS = {
        'beta': beta,
        'z_mean': z_mean,
        'z_bars': z_bars,
        's_z': s_z,
    }

    return MODEL_STATS


def predict_hkeaa_model(df, model_stats):
    """Apply the trained model to calculate moderated GPA marks."""
    # Extract statistics
    beta = model_stats['beta']
    X_mean = df['Raw College GPA (X)'].mean()
    x_bars = df.groupby('College')['Raw College GPA (X)'].mean()
    z_mean = model_stats['z_mean']
    z_bars = model_stats['z_bars']
    s_x = df.groupby('College')['Raw College GPA (X)'].std()
    s_z = model_stats['s_z']
    s_p = np.sqrt(0.5 * s_x**2 + 0.5 * s_z**2)
    
    # Map group-level statistics to the DataFrame
    df['z_bar'] = df['College'].map(z_bars)
    df['s_p'] = df['College'].map(s_p)
    df['s_x'] = df['College'].map(s_x)
    df['x_bar'] = df['College'].map(x_bars)

    # Apply the moderation formula
    df['Moderated GPA (Y)'] = (
        X_mean +
        beta * (df['z_bar'] - z_mean) +
        (df['Raw College GPA (X)'] - df['x_bar']) * (df['s_p'] / df['s_x'])
    )
    df['Moderated GPA (Y)'] = df['Moderated GPA (Y)'].clip(lower=0)
    return df


def train_mean_adjust_model(df):
    """Train the mean adjustment model and return the calculated statistics."""
    MODEL_STATS = {}
    
    # Compute required statistics
    x_bars = df.groupby('College')['Raw College GPA (X)'].mean()
    df['x_bar'] = df['College'].map(x_bars)
    z_bars = df.groupby('College')['University Performance GPA (Z)'].mean()
    df['z_bar'] = df['College'].map(z_bars)

    # Apply the mean adjustment formula
    df['Moderated GPA (Y)'] = (
        df['z_bar'] + (df['Raw College GPA (X)'] - df['x_bar'])
    )
    # Store model parameters
    MODEL_STATS = {
        'z_bars': z_bars,
    }

    return MODEL_STATS


def predict_mean_adjust_model(df, model_stats):
    """Apply the mean adjustment model to calculate moderated GPA marks."""
    # Extract statistics
    z_bars = model_stats['z_bars']
    df['z_bar'] = df['College'].map(z_bars)
    x_bars = df.groupby('College')['Raw College GPA (X)'].mean()
    df['x_bar'] = df['College'].map(x_bars)

    # Apply the mean adjustment formula
    df['Moderated GPA (Y)'] = (
        df['z_bar'] + (df['Raw College GPA (X)'] - df['x_bar'])
    )
    return df


def train_hkeaa_model_1(df):
    """Train the moderation model and return the calculated statistics."""
    MODEL_STATS = {}
    
    # Train regression model for beta
    model = LinearRegression()
    model.fit(df[['University Performance GPA (Z)']], df['Raw College GPA (X)'])
    beta = model.coef_[0]

    # Compute required statistics
    X_mean = df['Raw College GPA (X)'].mean()
    z_mean = df['University Performance GPA (Z)'].mean()
    z_bars = df.groupby('College')['University Performance GPA (Z)'].mean()
    x_bars = df.groupby('College')['Raw College GPA (X)'].mean()
    s_x = df.groupby('College')['Raw College GPA (X)'].std()
    s_z = df.groupby('College')['University Performance GPA (Z)'].std()

    # Map group-level statistics to the DataFrame
    df['z_bar'] = df['College'].map(z_bars)
    df['x_bar'] = df['College'].map(x_bars)
    df['s_z'] = df['College'].map(s_z)
    df['s_x'] = df['College'].map(s_x)

    # Calculate moderated GPA scores
    df['Moderated GPA (Y)'] = (
        X_mean +
        beta * (df['z_bar'] - z_mean) +
        (df['Raw College GPA (X)'] - df['x_bar']) * (df['s_z'] / df['s_x'])
    )

    # Store model parameters
    MODEL_STATS = {
        'beta': beta,
        'z_mean': z_mean,
        'z_bars': z_bars,
        's_z': s_z,
    }

    return MODEL_STATS


def predict_hkeaa_model_1(df, model_stats):
    """Apply the trained model to calculate moderated GPA marks."""
    # Extract statistics
    beta = model_stats['beta']
    X_mean = df['Raw College GPA (X)'].mean()
    x_bars = df.groupby('College')['Raw College GPA (X)'].mean()
    z_mean = model_stats['z_mean']
    z_bars = model_stats['z_bars']
    s_x = df.groupby('College')['Raw College GPA (X)'].std()
    s_z = model_stats['s_z']
    
    # Map group-level statistics to the DataFrame
    df['z_bar'] = df['College'].map(z_bars)
    df['s_z'] = df['College'].map(s_z)
    df['s_x'] = df['College'].map(s_x)
    df['x_bar'] = df['College'].map(x_bars)

    # Calculate moderated GPA scores
    df['Moderated GPA (Y)'] = (
        X_mean +
        beta * (df['z_bar'] - z_mean) +
        (df['Raw College GPA (X)'] - df['x_bar']) * (df['s_z'] / df['s_x'])
    )
    st.dataframe(df.describe())
    return df