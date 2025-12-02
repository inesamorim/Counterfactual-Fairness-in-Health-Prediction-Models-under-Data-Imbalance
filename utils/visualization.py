import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def plot_feature_distributions(df, features, target=None, figsize=(12, 6)): 
    """ Plots the distribution of multiple features in subplots (2 per row). 
    Parameters: ----------- 
    df : pandas.DataFrame The dataset. 
    features : list List of column names to plot. 
    target : str, optional 
    Target variable for hue separation (for categorical comparison). 
    figsize : tuple, optional Overall figure size. """ 
    num_features = len(features) 
    rows = (num_features + 1) / 3 # two plots per row 
    rows = round(rows) 
    fig, axes = plt.subplots(rows, 3, figsize=(figsize[0], figsize[1] * rows)) 
    axes = axes.flatten() 
    for i, feature in enumerate(features): 
        ax = axes[i] 
        # Choose plot type based on data type 
        if df[feature].dtype == 'object' or df[feature].nunique() < 10: 
            # Categorical variable 
            sns.countplot(data=df, x=feature, hue=feature, legend=False, ax=ax, palette="Set2") 
        else: 
            # Continuous variable 
            if target: sns.kdeplot(data=df, x=feature, hue=feature, legend=False, ax=ax, fill=True, common_norm=False, palette="Set2") 
            else: 
                sns.histplot(df[feature], ax=ax, kde=True, color="skyblue") 
                ax.set_title(f"Distribution of {feature}", fontsize=12) 
                ax.set_xlabel(feature) 
                ax.set_ylabel("Count" if df[feature].dtype == 'object' else "Density") 
                ax.grid(axis='y', linestyle='--', alpha=0.5) 
                # Hide unused subplots 
                for j in range(i + 1, len(axes)): fig.delaxes(axes[j]) 
                plt.tight_layout() 
                plt.show()

def plot_feature_distribution_categorical(df, feature, target=None, bins=30):
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
    plt.figure(figsize=(10, 5))
    if target:
        sns.histplot(data=df, x=feature, hue=target, bins=bins, kde=True, palette='Set2', element='step')
        plt.title(f'Distribution of {feature} by {target}')
    else:
        sns.histplot(data=df, x=feature, bins=bins, kde=True, color='green', element='step')
        plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, method='pearson', figsize=(10, 8), annot=True):
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        raise ValueError("No numeric columns found for correlation matrix.")

    corr = numeric_df.corr(method=method)
    plt.figure(figsize=figsize)
    sns.set(style="whitegrid", font_scale=0.9)
    ax = sns.heatmap(
        corr, annot=annot, fmt=".2f", cmap='coolwarm', center=0, linewidths=0.5,
        annot_kws={"size": 8, "color": "black"}, cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'}
    )
    plt.title(f'{method.capitalize()} Correlation Matrix', fontsize=14, pad=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def compare_categorical_distribution(df, feature, target):
    """
    Compare the distribution of a categorical feature with the target variable.

    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature.
        target (str): The name of the target variable.
    """
    # Check if feature and target exist
    if feature not in df.columns or target not in df.columns:
        raise ValueError("Feature or target not found in dataframe.")

    # Check that feature is categorical
    #if not (df[feature].dtype == 'object' or str(df[feature].dtype).startswith('category')):
    #    raise TypeError(f"Feature '{feature}' should be categorical, got {df[feature].dtype}")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=feature, hue=target, palette='Set2')
    plt.title(f"Distribution of {feature} by {target}", fontsize=14)
    plt.xlabel(feature.capitalize())
    plt.ylabel("Count")
    plt.legend(title=target)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()


def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def categorical_assoc_with_target(df, categorical_features, target):
    associations = {}
    for feature in categorical_features:
        confusion_mat = pd.crosstab(df[feature], df[target])
        associations[feature] = cramers_v(confusion_mat)
    
    assoc_df = pd.DataFrame.from_dict(associations, orient='index', columns=['Cramér_V'])
    assoc_df = assoc_df.sort_values(by='Cramér_V', ascending=False)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=assoc_df['Cramér_V'], y=assoc_df.index, palette='viridis')
    plt.title('Cramér’s V Association with Target')
    plt.xlabel('Strength of Association')
    plt.ylabel('Categorical Feature')
    plt.show()
    
    return assoc_df

def plot_numerical_distributions(df, num_features, target_col='HasHeartDisease'):
    """
    Plots the distribution of heart disease across numerical features.

    Parameters:
    - df: pandas DataFrame
    - num_features: list of numerical feature names
    - target_col: target column name (default = 'HasHeartDisease')
    """

    n_features = len(num_features)
    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(num_features):
        sns.kdeplot(
            data=df,
            x=feature,
            hue=target_col,
            fill=True,
            common_norm=False,
            palette='Set2',
            ax=axes[i]
        )
        axes[i].set_title(f'Distribution of {feature} by {target_col}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Density')

    # Remove empty subplots if any
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

  
