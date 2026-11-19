    assignments = rng.uniform(40, 100, n)
    score = (
        5.5 * study_hours
        + 0.28 * attendance
        + 0.18 * assignments
        + rng.normal(0, 6, n)
    )
    score = np.clip(score, 0, 100)

    df = pd.DataFrame({
        "study_hours": study_hours.round(2),
        "attendance": attendance.round(2),
        "assignment_score": assignments.round(2),
        "final_score": score.round(2),
    })

    # Demonstrate missing-value handling
    df.loc[[5, 17, 44], "attendance"] = np.nan
    return df


def pandas_numpy_practice(df):
    print("\n=== Pandas & NumPy ===")
    print("First rows:\n", df.head())
    print("\nDataset info:")
    print(df.info())
    print("\nSummary statistics:\n", df.describe())
    print("\nMissing values:\n", df.isna().sum())

    clean = df.copy()
    clean["attendance"] = clean["attendance"].fillna(clean["attendance"].median())
    clean = clean.drop_duplicates()

    # NumPy vectorized operation: categorize scores
    clean["score_category"] = np.where(
        clean["final_score"] >= 75, "High",
        np.where(clean["final_score"] >= 50, "Medium", "Low")
    )

    # Filtering and grouping
    high_performers = clean[clean["final_score"] >= 75]
    print("\nHigh performers:", len(high_performers))
    print("\nAverage score by category:\n",
          clean.groupby("score_category")["final_score"].mean().round(2))

    clean.to_csv(OUTPUT_DIR / "practice_dataset_cleaned.csv", index=False)
    return clean


def visualization_practice(df):
    print("\n=== Matplotlib & Seaborn ===")
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="final_score", kde=True)
    plt.title("Distribution of Final Scores")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "score_distribution.png", dpi=150)
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="study_hours", y="final_score")
    plt.title("Study Hours vs Final Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "study_vs_score.png", dpi=150)
    plt.show()

    plt.figure(figsize=(7, 5))
    sns.heatmap(df.select_dtypes(include="number").corr(),
                annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.show()


def sklearn_practice(df):
    print("\n=== Scikit-learn ===")
    features = ["study_hours", "attendance", "assignment_score"]
    X = df[features]
    y = df["final_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.3f}")

    results = pd.DataFrame({
        "actual_score": y_test.values,
        "predicted_score": predictions.round(2)
    })
    results.to_csv(OUTPUT_DIR / "model_predictions.csv", index=False)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=predictions)
    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.xlabel("Actual Scores")
    plt.ylabel("Predicted Scores")
    plt.title("Actual vs Predicted Scores")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.show()


def main():
    print("Data Science Tool Mastery: Practical Demonstrations")
    data = create_sample_data()
    processed = pandas_numpy_practice(data)
    visualization_practice(processed)
    sklearn_practice(processed)
    print(f"\nFinished! Outputs saved in: {OUTPUT_DIR.resolve()}")
