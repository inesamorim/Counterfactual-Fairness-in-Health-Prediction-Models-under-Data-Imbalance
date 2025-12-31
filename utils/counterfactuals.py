import dice_ml
from dice_ml import Dice
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def find_counterfactuals(model, df, X_test, y_test, sex, features_to_vary=None, total_CFs=5):
    """
    Generate counterfactuals for samples in X_test filtered by sex using DiCE.
    Ensures that only counterfactuals that actually flip the prediction are kept.
    """
    if features_to_vary is None:
        features_to_vary = ["sex", "age", "chol", "thalach", "oldpeak", 'ca', "trestbps"]

    # --- Clean df and X_test ---
    target_col = "HasHeartDisease"
    for col in [target_col] + features_to_vary:
        df = df.dropna(subset=[col])
    for col in features_to_vary:
        X_test = X_test.dropna(subset=[col])

    df[target_col] = df[target_col].astype(int)

    model_dice = dice_ml.Model(
        model=model,
        backend="sklearn",
        model_type="classifier"
    )

    data_dice = dice_ml.Data(
        dataframe=df,
        continuous_features=[
           "sex", "age", "chol", "thalach", "oldpeak", 'ca', "trestbps"
        ],
        categorical_features=[
            "sex", "cp_2", "cp_3", "cp_4", "fbs",
            "restecg_1", "restecg_2", "exang",
            "slope_2", "slope_3", "thal_6.0", "thal_7.0"
        ],
        outcome_name=target_col
    )

    exp = Dice(data_dice, model_dice, method="random")
    samples = X_test[X_test["sex"] == sex]

    results = []
    failed_samples = 0
    sample_id = 0

    for idx, row in samples.iterrows():
        query = pd.DataFrame([row])

        try:
            cf = exp.generate_counterfactuals(
                query,
                total_CFs=total_CFs,
                desired_class="opposite",
                features_to_vary=features_to_vary,
                verbose=False,
                random_seed=42
            )

            # ensure CFs were produced by DiCE
            if not getattr(cf, "cf_examples_list", None):
                failed_samples += 1
                print(f"No cf_examples_list for idx {idx}, skipping.")
                continue

            # --- counterfactual(s) ---
            cf_df = cf.cf_examples_list[0].final_cfs_df.copy()
            cf_df["type"] = "counterfactual"

            # Filter: keep only counterfactuals that actually flip the prediction
            orig_y = int(y_test.loc[idx])
            cf_df = cf_df[cf_df[target_col] != orig_y]
            print(f"Candidate for idx {idx}: Found {len(cf_df)} valid counterfactual(s) that flip the prediction.")

            if not cf_df.empty:
                # --- original instance (only add when CFs exist) ---
                original = query.copy()
                original["type"] = "original"
                original["sample_id"] = sample_id
                original[target_col] = orig_y
                results.append(original)

                cf_df["sample_id"] = sample_id
                results.append(cf_df)
                sample_id += 1
            else:
                failed_samples += 1

        except Exception as e:
            failed_samples += 1
            print(f"Error generating CF for idx {idx}: {e}")
            continue

    if results:
        print(
            f"Generated counterfactuals for {sample_id} samples, "
            f"skipped {failed_samples} samples."
        )
        return pd.concat(results, ignore_index=True)

    print("No counterfactuals found")
    return None



def compute_cf_deltas(cf_df, features=None):
    """Compute CF deltas (CF - original) for specified features."""
    deltas = []
    if features is None:
        features = ["sex", "age", "chol", "thalach", "oldpeak", 'ca', "trestbps"]

    for sid in cf_df["sample_id"].unique():
        #print(f"Computing deltas for sample_id {sid}")
        #print(cf_df[cf_df["sample_id"] == sid])
        original = cf_df[(cf_df["sample_id"] == sid) & (cf_df["type"] == "original")]
        counterfactual = cf_df[(cf_df["sample_id"] == sid) & (cf_df["type"] == "counterfactual")]

        if original.empty or counterfactual.empty:
            print(f"Skipping sample_id {sid} due to missing original or counterfactual.")
            continue

        delta = counterfactual[features].values - original[features].values
        delta_df = pd.DataFrame(delta, columns=features)
        #print(f"Deltas for sample_id {sid}:\n{delta_df}")

        if len(delta_df) != len(cf_df[cf_df["sample_id"] == sid]) - 1:
            print(f"Warning: Mismatch in number of deltas and counterfactuals for sample_id {sid}")
            print(f"Number of deltas: {len(delta_df)}, Number of counterfactuals: {len(cf_df[cf_df['sample_id'] == sid]) - 1}")


        flip = cf_df[(cf_df["sample_id"] == sid) & (cf_df["type"] == "counterfactual")]["flip"].values[0]
        delta_df["flip"] = flip
        delta_df["sample_id"] = sid
        deltas.append(delta_df)

    return pd.concat(deltas, ignore_index=True)


def add_flip_direction(cf_df, target_col="HasHeartDisease"):
    """
    Adds a 'flip' column indicating prediction change:
    - '0→1'
    - '1→0'
    """
    for sid in cf_df["sample_id"].unique():
        original = cf_df[(cf_df["sample_id"] == sid) & (cf_df["type"] == "original")]
        cf = cf_df[(cf_df["sample_id"] == sid) & (cf_df["type"] == "counterfactual")]

        if original.empty or cf.empty:
            continue

        orig_y = int(original[target_col].iloc[0])
        cf_y = int(cf[target_col].iloc[0])
        flip = f"{orig_y}→{cf_y}"

        cf_df.loc[cf_df["sample_id"] == sid, "flip"] = flip
        cf_df.loc[((cf_df["sample_id"] == sid) & (cf_df["type"] == "original")), "flip"] = f"{orig_y}→{orig_y}"

    return cf_df


def plot_cf_distributions(cf_results, features_to_plot=None):
    """Plot heatmaps of counterfactual feature changes."""
    if features_to_plot is None:
        features_to_plot = ["sex", "age", "chol", "thalach", "oldpeak", 'ca', "trestbps"]

    cf_results = add_flip_direction(cf_results, target_col="HasHeartDisease")
    delta_df = compute_cf_deltas(cf_results, features_to_plot)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    #print(cf_results['flip'].value_counts())
    #print(delta_df['flip'].value_counts())
    for ax, flip in zip(axes, ["0→1", "1→0"]):

        subset = delta_df[delta_df["flip"] == flip]
        if subset.empty:
            print(f"No data for flip direction {flip}, skipping plot.")
            continue

        sns.heatmap(
            subset[features_to_plot],
            cmap="coolwarm",
            center=0,
            cbar=True,
            ax=ax
        )
        ax.set_title(f"Counterfactual Feature Changes ({flip})")
        ax.set_xlabel("Features")
        ax.set_ylabel("Instances")

    plt.tight_layout()
    plt.show()

    return delta_df


def plot_cf_boxplots(cf_results, features_to_plot=None):
    """Plot boxplots of counterfactual feature changes."""
    if features_to_plot is None:
        features_to_plot = ["sex", "age", "chol", "thalach", "oldpeak", 'ca', "trestbps"]

    cf_results = add_flip_direction(cf_results, target_col="HasHeartDisease")
    delta_df = compute_cf_deltas(cf_results, features_to_plot)
    #delta_long = delta_df.melt(
    #    id_vars="sample_id",
    #    value_vars=features_to_plot,
    #    var_name="feature",
    #    value_name="delta"
    #)
    delta_df = delta_df.melt(
        id_vars=["sample_id", "flip"] if "flip" in delta_df else ["sample_id"],
        value_vars=features_to_plot,
        var_name="feature",
        value_name="delta"
    )
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, flip in zip(fig.axes, ["0→1", "1→0"]):
        subset = delta_df[delta_df["flip"] == flip]
        if subset.empty:
            print(f"No data for flip direction {flip}, skipping plot.")
            continue

        
        sns.boxplot(data=subset, x="feature", y="delta", ax=ax, color='lightgreen')
        ax.axhline(0, color="black", linestyle="--")
        ax.set_title(f"Distribution of Feature Changes (CF − Original) ({flip})")
        ax.set_ylabel("Delta")  
    plt.tight_layout()
    plt.show()

def plot_gender_flips(cf_df):
    """
    Visualize how the 'sex' feature changes in counterfactuals.
    cf_df must include columns: 'sample_id', 'type' ('original'/'counterfactual'), 'sex'
    """

    cf_df = add_flip_direction(cf_df, target_col="HasHeartDisease")
    
    # For each sample, check if sex changed
    sex_flips = []
    for sid in cf_df['sample_id'].unique():
        original = cf_df[(cf_df['sample_id'] == sid) & (cf_df['type'] == 'original')]['sex'].iloc[0]
        cf = cf_df[(cf_df['sample_id'] == sid) & (cf_df['type'] == 'counterfactual')]['sex'].iloc[0]
        flip_direction = cf_df[(cf_df['sample_id'] == sid) & (cf_df['type'] == 'counterfactual')]['flip'].iloc[0]
        sex_flips.append({'sample_id': sid, 'sex_original': original, 'sex_cf': cf, 'sex_changed': original != cf, 'flip_direction': flip_direction})

    sex_flips_df = pd.DataFrame(sex_flips)
    sex_flips_df = sex_flips_df[
    (sex_flips_df['flip_direction'] == "0→1") |
    (sex_flips_df['flip_direction'] == "1→0")
    ]
    percentage_changed = sex_flips_df['sex_changed'].mean() * 100
    print(f"Percentage of counterfactuals where gender changed: {percentage_changed:.2f}%")


    
    # Plot
    plt.figure(figsize=(6,4))
    sns.countplot(data=sex_flips_df, x='sex_changed', palette=['lightblue', 'salmon'], hue='sex_changed', legend=False)
    plt.xticks([0,1], ['No Change', 'Changed'])
    plt.ylabel("Number of Counterfactuals")
    plt.title("Gender Changes in Counterfactuals (from original males)")
    plt.show()
    
    return sex_flips_df[sex_flips_df['sex_changed']==True]

def show_mean_delta(delta_df, mapping, scaler):
    # mean changes in features for counterfactuals
    mean_changes = {}
    for feature in ["age", "chol", "thalach", "oldpeak", 'ca', "trestbps"]:
        mean_change = pd.DataFrame(delta_df.groupby('flip')[feature].mean())
        mean_change = mean_change.reset_index()
        mean_change = mean_change.rename(columns={feature: 'mean_change'})
        #only keep rows where flip is "0→1" or "1→0"
        mean_change = mean_change[(mean_change['flip'] == "0→1") | (mean_change['flip'] == "1→0")]
        mean_change['std'] = pd.DataFrame(delta_df.groupby('flip')[feature].std()).reset_index()[feature]
        mean_changes[feature] = mean_change

    mean_changes_descaled = {}
    for feature in ["age", "chol", "thalach", "oldpeak", 'ca', "trestbps"]:
        mean_change = mean_changes[feature].copy()
        mean_change['mean_change_descaled'] = mean_change['mean_change'] * scaler.scale_[mapping[feature]]
        mean_change['std_descaled'] = mean_change['std'] * scaler.scale_[mapping[feature]]
        mean_changes_descaled[feature] = mean_change

        print(f'Mean change in {feature} (descaled):\n{mean_change}\n')


def descale_and_inspect(
    df,
    scaler,
    continuous_features,
    round_decimals=2
):
    """
    De-scales continuous features and returns a readable DataFrame
    with all features preserved.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing scaled features (e.g. counterfactuals).
    scaler : fitted scaler (e.g. StandardScaler)
        Must have mean_ and scale_ attributes.
    continuous_features : list[str]
        Names of continuous features to de-scale.
    round_decimals : int
        Decimal rounding for readability.

    Returns
    -------
    pd.DataFrame
        DataFrame with continuous features de-scaled and others unchanged.
    """

    df_out = df.copy()
    #print(df[continuous_features].columns)

    # Sanity check
    missing = set(continuous_features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing continuous features in df: {missing}")

    # De-scale
    df_out[continuous_features] = scaler.inverse_transform(df_out[continuous_features])

    # Round continuous features for readability
    df_out[continuous_features] = df_out[continuous_features].round(round_decimals)

    return df_out
