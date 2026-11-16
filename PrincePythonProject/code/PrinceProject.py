"""
Electricity Usage Predictor
---------------------------
Loads past monthly electricity usage from a CSV file (columns: Month, Units),
fits a linear regression line with NumPy (Units = slope * Month + intercept),
predicts units for a future month, estimates the bill and plots the results.
"""

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RATE_PER_UNIT = 6  # Rs. per unit
MIN_ROWS = 2       # minimum rows required for regression


# ---------------------------------------------------------------- loading
def load_data(path):
    """Load CSV, validate it and return a clean DataFrame (or None on error)."""
    try:
        # header=None so files with or without a header row both work
        df = pd.read_csv(path, header=None, skipinitialspace=True)
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        return None
    except Exception as exc:
        print(f"Error: Could not read the CSV file ({exc}).")
        return None

    if df.shape[1] < 2:
        print("Error: CSV must contain two columns: Month and Units.")
        return None

    df = df.iloc[:, :2]
    df.columns = ["Month", "Units"]

    # If the first row is a header (e.g. "Month,Units"), drop it
    first = pd.to_numeric(df.iloc[0], errors="coerce")
    if first.isna().all():
        df = df.iloc[1:].reset_index(drop=True)

    # Convert to numbers; invalid entries become NaN
    df = df.apply(pd.to_numeric, errors="coerce")

    # Check for missing / non-numeric values
    bad = int(df.isna().sum().sum())
    if bad:
        print(f"Warning: {bad} missing/non-numeric value(s) found. "
              "Affected rows were removed.")
        df = df.dropna().reset_index(drop=True)

    return df


def show_summary(df):
    """Print a short summary of the data."""
    print("\n--- Data Summary ---")
    print(f"Rows            : {len(df)}")
    print(f"Missing values  : {int(df.isna().sum().sum())}")
    if len(df) > 0:
        print(f"Months          : {int(df['Month'].min())} to {int(df['Month'].max())}")
        print(f"Units (min/avg/max): {df['Units'].min():.2f} / "
              f"{df['Units'].mean():.2f} / {df['Units'].max():.2f}")
    print("--------------------")


# ------------------------------------------------------------- regression
def fit_model(df):
    """Fit Units = slope * Month + intercept; return slope, intercept, R2."""
    x = df["Month"].to_numpy(dtype=float)
    y = df["Units"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot

    return slope, intercept, round(r2, 2)


def predict_units(slope, intercept, month):
    return slope * month + intercept


def estimate_bill(units):
    return units * RATE_PER_UNIT


# ---------------------------------------------------------------- plotting
def plot_results(df, slope, intercept, month, predicted):
    """Plot actual points, regression line and the predicted point."""
    x = df["Month"].to_numpy(dtype=float)
    y = df["Units"].to_numpy(dtype=float)

    x_line = np.linspace(min(x.min(), month), max(x.max(), month), 100)
    y_line = slope * x_line + intercept

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="blue", label="Actual Usage", zorder=3)
    plt.plot(x_line, y_line, color="red", label="Regression Line")
    plt.scatter([month], [predicted], color="green", marker="*", s=180,
                label=f"Predicted (Month {month:g})", zorder=4)
    plt.title("Actual vs Predicted")
    plt.xlabel("Month")
    plt.ylabel("Units Consumed")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png")
    plt.show()


# -------------------------------------------------------------------- menu
def read_month():
    """Ask for a month number; return float or None if invalid."""
    raw = input("Enter month number to predict: ").strip()
    try:
        month = float(raw)
    except ValueError:
        print("Error: Month must be a number.")
        return None
    if month <= 0:
        print("Error: Month must be a positive number.")
        return None
    return month


def predict_option(df):
    if len(df) < MIN_ROWS:
        print("Error: At least 2 months of data are required for prediction.")
        return
    if df["Month"].nunique() < 2:
        print("Error: At least 2 different months are required for prediction.")
        return

    month = read_month()
    if month is None:
        return

    slope, intercept, r2 = fit_model(df)
    predicted = predict_units(slope, intercept, month)
    bill = estimate_bill(predicted)

    print(f"\nSlope: {slope:.2f} | Intercept: {intercept:.2f} | R2: {r2:.2f}")
    print(f"Predicted units (Month {month:g}): {predicted:.2f}")
    print(f"Estimated bill: Rs. {bill:.2f}")

    plot_results(df, slope, intercept, month, predicted)


def main():
    path = input("Enter CSV file path (default: usage.csv): ").strip() or "usage.csv"
    df = load_data(path)
    if df is None:
        sys.exit(1)

    show_summary(df)

    while True:
        print("\n===== Electricity Usage Predictor =====")
        print("1. Predict")
        print("2. Show data summary")
        print("3. Exit")
        choice = input("Choose an option: ").strip()

        if choice == "1":
            predict_option(df)
        elif choice == "2":
            show_summary(df)
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please choose 1, 2 or 3.")


if __name__ == "__main__":
    main()