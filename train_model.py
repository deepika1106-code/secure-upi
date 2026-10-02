import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

FEATURES = ["amount", "hour", "new_beneficiary", "device_changed", "distance_km", "txns_last_hour"]
rng = np.random.default_rng(42)
n = 8000
df = pd.DataFrame({
    "amount": rng.gamma(2, 3000, n).clip(10, 100000).round(2),
    "hour": rng.integers(0, 24, n),
    "new_beneficiary": rng.integers(0, 2, n),
    "device_changed": (rng.random(n) < 0.15).astype(int),
    "distance_km": rng.exponential(30, n).round(1),
    "txns_last_hour": rng.poisson(1.0, n),
})
z = (-5.2 + 0.00006*df.amount + 2.0*df.new_beneficiary + 1.8*df.device_changed
     + 0.012*df.distance_km + 0.6*df.txns_last_hour + 1.5*(df.hour < 5))
df["is_fraud"] = ((z + rng.normal(0, 0.7, n)) > 0).astype(int)
df.to_csv("upi_transactions.csv", index=False)
print("Fraud rate: %.1f%%" % (df.is_fraud.mean()*100))

X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], df.is_fraud, test_size=0.2,
                                          stratify=df.is_fraud, random_state=42)
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
}
best, best_f1 = None, -1
for name, m in models.items():
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    f1 = f1_score(y_te, pred)
    print("\n==", name, "F1 = %.3f" % f1); print(classification_report(y_te, pred))
    if f1 > best_f1: best, best_f1, best_name = m, f1, name
joblib.dump(best, "model.pkl")
print("Saved best model:", best_name)
