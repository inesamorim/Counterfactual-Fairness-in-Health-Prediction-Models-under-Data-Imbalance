import dice_ml
from dice_ml import Dice
import pandas as pd
import contextlib
import io

def find_counterfactuals(model, df, X_test, sex):
    model_dice = dice_ml.Model(model=model, backend='sklearn', model_type='classifier')

    data_dice = dice_ml.Data(
        dataframe=df,
        continuous_features=['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca'],
        categorical_features=['sex', 'cp_2', 'cp_3', 'cp_4', 'fbs', 'restecg_1', 'restecg_2', 'exang', 'slope_2', 'slope_3', 'thal_6.0', 'thal_7.0'],
        outcome_name='HasHeartDisease'
    )

    exp = Dice(data_dice, model_dice, method='random')
    samples = X_test[X_test['sex'] == sex] # Filter samples by sex

    failed_samples = 0
    cf_results = []
    for i, row in samples.iterrows():
        query = pd.DataFrame([row])
        try:
            # suppress DiCE prints
            with contextlib.redirect_stdout(io.StringIO()):
                cf = exp.generate_counterfactuals(
                    query,
                    total_CFs=1,
                    desired_class="opposite",
                    features_to_vary=['sex'],
                    verbose=False
                )
            # Visualize counterfactual explanation
            cf.visualize_as_dataframe()
            cf_results.append(cf.cf_examples_list[0].final_cfs_df)
        except Exception as e:
            #print(f"Skipped sample {i}: {e}")
            failed_samples += 1
            continue

    if cf_results:
        print(f"Generated counterfactuals for {len(cf_results)} samples, skipped {failed_samples} samples.")
        return pd.concat(cf_results, ignore_index=True)
    else:
        print("No counterfactuals found")
        return None
