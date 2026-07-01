"""
Reproduce Table IV/V and Figs. 7, 8, 9 of the paper: binary classifier
for proteinoid spike prediction.

Train on Datasets 1-4, test on Dataset 5.

Run from the repository root:
    python scripts/04_train_classifier.py
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from proteinoid_spikes import config
from proteinoid_spikes.classifier import (
    create_model, evaluate_model, set_global_seed, train_model,
)
from proteinoid_spikes.data_loading import load_all_datasets
from proteinoid_spikes.features import engineer_features, get_feature_columns
from proteinoid_spikes.plotting import (
    plot_confusion_matrix, plot_roc_curve, plot_training_history,
)


def main() -> None:
    set_global_seed()

    dataframes = load_all_datasets()

    # Split into train (Datasets 1-4) and test (Dataset 5) by Dataset column.
    train_dfs = [df for df in dataframes if df["Dataset"].iloc[0] in config.TRAIN_DATASETS]
    test_dfs  = [df for df in dataframes if df["Dataset"].iloc[0] in config.TEST_DATASETS]

    dataframe_1 = pd.concat(train_dfs, ignore_index=True)
    dataframe_2 = test_dfs[0]

    # Feature engineering (16-dim vector)
    dataframe_1 = engineer_features(dataframe_1)
    dataframe_2 = engineer_features(dataframe_2)

    feature_columns = get_feature_columns(dataframe_1)
    X_train = dataframe_1[feature_columns].values
    y_train = dataframe_1["Spike"].values
    X_test = dataframe_2[feature_columns].values
    y_test = dataframe_2["Spike"].values

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model = create_model(X_train_scaled.shape[1])
    history = train_model(model, X_train_scaled, y_train)

    # Evaluate
    results = evaluate_model(model, X_test_scaled, y_test)

    # Print Table IV/V
    print("\nClassification Performance Metrics:")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"AUC:       {results['auc']:.4f}")

    cm = results["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives:  {tn}")
    print(f"False Negatives: {fn}")

    # Save Table IV/V as CSV
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_table = pd.DataFrame([{
        "True Positives": int(tp),
        "False Positives": int(fp),
        "True Negatives": int(tn),
        "False Negatives": int(fn),
        "Accuracy": results["accuracy"],
        "Precision": results["precision"],
        "Recall": results["recall"],
        "F1": results["f1"],
        "AUC": results["auc"],
    }])
    metrics_table_path = config.TABLES_DIR / "table4_classification_metrics.csv"
    metrics_table.to_csv(metrics_table_path, index=False)
    print(f"\nSaved: {metrics_table_path}")

    # Save predictions for Dataset 5 (as in the original notebook)
    dataframe_2 = dataframe_2.copy()
    dataframe_2["Predicted_Spike"] = results["y_pred"]
    predictions_path = config.TABLES_DIR / "dataset5_predictions.csv"
    dataframe_2[["Time", "Spike", "Predicted_Spike"]].to_csv(predictions_path, index=False)
    print(f"Saved: {predictions_path}")

    # Save figures
    fig7_path = config.FIGURES_DIR / "fig7_confusion_matrix.png"
    plot_confusion_matrix(cm, save_path=fig7_path)
    print(f"Saved: {fig7_path}")

    fig8_path = config.FIGURES_DIR / "fig8_training_history.png"
    plot_training_history(history, save_path=fig8_path)
    print(f"Saved: {fig8_path}")

    fig9_path = config.FIGURES_DIR / "fig9_roc_curve.png"
    plot_roc_curve(results["fpr"], results["tpr"], results["auc"], save_path=fig9_path)
    print(f"Saved: {fig9_path}")


if __name__ == "__main__":
    main()