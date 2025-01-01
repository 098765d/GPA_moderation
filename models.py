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

def train_reg_std_model(df):
    """Train the moderation model and return the calculated statistics."""
    MODEL_STATS = {}

    # Train regression model for each college to get individual beta values
    colleges = df['College'].unique()
    beta_values = {}
    for college in colleges:
        college_data = df[df['College'] == college]
        model = LinearRegression()
        model.fit(college_data[['University Performance GPA (Z)']], college_data['Raw College GPA (X)'])
        beta_values[college] = model.coef_[0]

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
    df['beta'] = df['College'].map(beta_values)  # Map individual beta values

    # Calculate moderated GPA scores using college-specific beta values
    df['Moderated GPA (Y)'] = (
        X_mean +
        df['beta'] * (df['z_bar'] - z_mean) +
        (df['Raw College GPA (X)'] - df['x_bar']) * (df['s_z'] / df['s_x'])
    )

    # Store model parameters
    MODEL_STATS = {
        'beta_values': beta_values,
        'z_mean': z_mean,
        'z_bars': z_bars,
        's_z': s_z,
    }

    return MODEL_STATS

def predict_reg_std_model(df, model_stats):
    """Apply the trained model to calculate moderated GPA marks."""
    # Extract statistics
    beta_values = model_stats['beta_values']
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
    df['beta'] = df['College'].map(beta_values)  # Map individual beta values

    # Calculate moderated GPA scores using college-specific beta values
    df['Moderated GPA (Y)'] = (
        X_mean +
        df['beta'] * (df['z_bar'] - z_mean) +
        (df['Raw College GPA (X)'] - df['x_bar']) * (df['s_z'] / df['s_x'])
    )
    st.dataframe(df.describe())
    return df


def train_linear_regression_model(df):
    """
    Train a simple linear regression model where:
    Z = intercept + beta * X
    and store these parameters.
    
    Returns:
        MODEL_STATS (dict): Contains 'beta' and 'intercept' of the fitted model.
    """
    MODEL_STATS = {}
    
    # Train regression model for beta
    model = LinearRegression()
    model.fit(df[['Raw College GPA (X)']], df['University Performance GPA (Z)'])
    beta = model.coef_[0]
    intercept = model.intercept_

    # Store model parameters
    MODEL_STATS = {
        'k': beta,
        'intercept': intercept
    }

    return MODEL_STATS


def predict_linear_regression_model(df, model_stats):
    """
    Apply the trained linear regression model to calculate the moderated values 'yu'.
    
    Formula:
    y = intercept + beta * X
    
    Where 'X' is the Raw College GPA (X).
    """
    # Extract statistics
    beta = model_stats['k']
    intercept = model_stats['intercept']
    
    # Apply the moderation formula
    df['Moderated GPA (Y)'] = intercept + beta * df['Raw College GPA (X)']
    return df



