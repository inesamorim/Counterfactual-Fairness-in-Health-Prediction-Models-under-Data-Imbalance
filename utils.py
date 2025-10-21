import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_feature_distribution_categorical(df, feature, target=None, bins=30):
    """
    Plots the distribution of a feature
    Parameters:
    - df: pandas DataFrame
    - feature: str, column name to visualize
    - target: str, optional column name for grouping (e.g. 'sex' or 'target')
    - bins: int, number of bins for numeric histogram
    """
    plt.figure(figsize=(10, 5))
    if target:
        sns.countplot(data=df, x=feature, hue=target, palette='Set2')
        plt.title(f'Count of {feature} by {target}')
    else:
        sns.countplot(data=df, x=feature, palette='Set2')
        plt.title(f'Count of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_feature_distribution_numeric(df, feature, target=None, bins=30):
    """
    Plots the distribution of a numeric feature
    Parameters:
    - df: pandas DataFrame
    - feature: str, column name to visualize
    - target: str, optional column name for grouping
    - bins: int, number of bins for histogram
    """
    plt.figure(figsize=(10, 5))
    if target:
        sns.histplot(data=df, x=feature, hue=target, bins=bins, kde=True, palette='Set2', element='step')
        plt.title(f'Distribution of {feature} by {target}')
    else:
        sns.histplot(data=df, x=feature, bins=bins, kde=True, color='steelblue', element='step')
        plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
