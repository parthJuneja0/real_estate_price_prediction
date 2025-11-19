import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load


DATA_PATH = "data/data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "Dragon.joblib")


def load_housing_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find {DATA_PATH}. Make sure data.csv is placed in data/ folder.")
    housing = pd.read_csv(DATA_PATH)
    print("Columns in dataset:", list(housing.columns))
    return housing


def get_target_column_name(housing: pd.DataFrame) -> str:
    # Prefer the exact Y column name if present
    for col in housing.columns:
        if col.strip().lower().startswith("y"):
            return col
    # Fallback: assume last column is the target
    return housing.columns[-1]


def stratified_split_by_price(housing: pd.DataFrame, target_col: str):
    """
    Create a price category from the target and use it for stratified split.
    This keeps the distribution of house prices similar in train & test sets.
    """
    housing = housing.copy()

    # Create price categories based on quantiles of the target
    housing["price_cat"] = pd.qcut(housing[target_col], q=4, labels=[1, 2, 3, 4])

    print("\nPrice category value counts (for stratification):")
    print(housing["price_cat"].value_counts())

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(housing, housing["price_cat"]):
        strat_train_set = housing.loc[train_idx].drop("price_cat", axis=1)
        strat_test_set = housing.loc[test_idx].drop("price_cat", axis=1)

    return strat_train_set, strat_test_set


def explore_data(housing: pd.DataFrame, target_col: str):
    print("\n=== HEAD ===")
    print(housing.head())

    print("\n=== INFO ===")
    print(housing.info())

    print("\n=== DESCRIBE ===")
    print(housing.describe())

    # Simple scatter plot: distance to MRT vs price
    if "X3 distance to the nearest MRT station" in housing.columns:
        housing.plot(
            kind="scatter",
            x="X3 distance to the nearest MRT station",
            y=target_col,
            alpha=0.6,
        )
        plt.title("Distance to MRT vs Price")
        plt.show()
    
    # Simple scatter plot: number of convenience stores vs price
    if "X4 number of convenience stores" in housing.columns:
        housing.plot(
            kind="scatter",
            x="X4 number of convenience stores",
            y=target_col,
            alpha=0.6,
        )
        plt.title("Convenience stores vs Price")
        plt.show()


def prepare_data(train_df: pd.DataFrame, target_col: str):
    # Separate labels
    y = train_df[target_col].copy()
    X = train_df.drop(target_col, axis=1)

    # Drop ID-like column if present
    if "No" in X.columns:
        X = X.drop("No", axis=1)

    # Numeric pipeline: impute missing values + scale
    my_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("std_scaler", StandardScaler()),
    ])

    X_prepared = my_pipeline.fit_transform(X)
    return X, y, X_prepared, my_pipeline


def train_model(X_prepared, y):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_prepared, y)
    return model


def evaluate_model(model, X_prepared, y):
    # Training RMSE
    predictions = model.predict(X_prepared)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    print("\n=== Training set RMSE ===")
    print(rmse)

    # Cross-validation RMSE
    scores = cross_val_score(
        model,
        X_prepared,
        y,
        scoring="neg_mean_squared_error",
        cv=10
    )
    rmse_scores = np.sqrt(-scores)

    print("\n=== Cross-validation RMSE scores ===")
    print("Scores:", rmse_scores)
    print("Mean:  ", rmse_scores.mean())
    print("Std:   ", rmse_scores.std())


def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


def test_model_on_test_set(model, my_pipeline, test_df: pd.DataFrame, target_col: str):
    y_test = test_df[target_col].copy()
    X_test = test_df.drop(target_col, axis=1)
    if "No" in X_test.columns:
        X_test = X_test.drop("No", axis=1)

    X_test_prepared = my_pipeline.transform(X_test)
    final_predictions = model.predict(X_test_prepared)

    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)

    print("\n=== Test set RMSE ===")
    print(final_rmse)


def demo_use_saved_model():
    if not os.path.exists(MODEL_PATH):
        print("\nSaved model not found, skipping demo usage.")
        return

    print("\n=== Demo: using saved model on one test-like example ===")
    model = load(MODEL_PATH)

    # Example dummy feature vector with 6 features (X1–X6).
    # Replace these values with real ones if you want.
    example_features = np.array([[
        2013.5,   # X1 transaction date (example)
        10.0,     # X2 house age
        300.0,    # X3 distance to MRT
        5.0,      # X4 number of convenience stores
        24.97,    # X5 latitude
        121.54,   # X6 longitude
    ]])

    prediction = model.predict(example_features)
    print("Predicted price for example features:", prediction)

def test_random_example(model, my_pipeline, strat_test_set, target_col):
    # pick one random row from test set (this data was NOT used for training)
    sample = strat_test_set.sample(1).copy()

    true_price = sample[target_col].values[0]

    # drop target
    X_sample = sample.drop(columns=[target_col])

    # drop ID-like column to match training features
    if "No" in X_sample.columns:
        X_sample = X_sample.drop("No", axis=1)

    # apply same pipeline as training
    X_prepared = my_pipeline.transform(X_sample)
    predicted_price = model.predict(X_prepared)[0]

    print("\n=== Predicting a random unseen example ===")
    print("Input features:\n", X_sample)
    print(f"\nTrue price:      {true_price}")
    print(f"Predicted price: {predicted_price}")


def main():
    # Load data
    housing = load_housing_data()
    target_col = get_target_column_name(housing)
    print("Using target column:", target_col)

    # Stratified split on binned price
    strat_train_set, strat_test_set = stratified_split_by_price(housing, target_col)

    # Quick EDA on train set (optional, can be commented out)
    explore_data(strat_train_set.copy(), target_col)

    # Prepare data & pipeline
    X_train, y_train, X_train_prepared, my_pipeline = prepare_data(strat_train_set, target_col)

    # Train model
    model = train_model(X_train_prepared, y_train)

    # Evaluate model
    evaluate_model(model, X_train_prepared, y_train)

    # Save model
    save_model(model)

    # Evaluate on test set
    test_model_on_test_set(model, my_pipeline, strat_test_set, target_col)

    test_random_example(model, my_pipeline, strat_test_set, target_col)


if __name__ == "__main__":
    main()
