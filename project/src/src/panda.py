from joblib import load

model_dir = "C:/Users/HP/PycharmProjects/ML_final_project/models/"

try:
    log_model = load(model_dir + "log_model.joblib")
    adaboost_model = load(model_dir + "adaboost_model.joblib")
    pca_model = load(model_dir + "pca_model.joblib")
    kmeans_model = load(model_dir + "kmeans_model.joblib")
    scaler = load(model_dir + "scaler.joblib")

    print("✅ Models loaded successfully!")

    # Test Logistic Regression
    if hasattr(log_model, "coef_"):
        print(f"Logistic Regression Coefficients: {log_model.coef_}")
    else:
        print("⚠️ Logistic Regression Model is not trained properly.")

    # Test AdaBoost
    if hasattr(adaboost_model, "estimators_"):
        print(f"AdaBoost Model contains {len(adaboost_model.estimators_)} weak learners.")
    else:
        print("⚠️ AdaBoost Model is not trained properly.")

    # Test PCA
    if hasattr(pca_model, "explained_variance_ratio_"):
        print(f"PCA Explained Variance Ratio: {sum(pca_model.explained_variance_ratio_):.2%}")
    else:
        print("⚠️ PCA Model is not trained properly.")

    # Test K-Means
    if hasattr(kmeans_model, "cluster_centers_"):
        print(f"K-Means Cluster Centers (First 3 Rows):\n{kmeans_model.cluster_centers_[:3]}")
    else:
        print("⚠️ K-Means Model is not trained properly.")

except Exception as e:
    print(f"❌ Error loading models: {e}")
