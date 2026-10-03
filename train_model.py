import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

FEATURES = ["amount", "hour", "receiver_new", "receiver_flags", "kw_hits", "digit_ratio",
            "unknown_handle", "txns_last_hour", "amount_ratio", "device_changed", "distance_km"]
rng = np.random.default_rng(42)
n = 10000
df = pd.DataFrame({
    "amount": rng.gamma(2, 3000, n).clip(10, 100000).round(2),
    "hour": rng.integers(0, 24, n),
    "receiver_new": (rng.random(n) < 0.5).astype(int),
    "receiver_flags": rng.poisson(0.2, n).clip(0, 5),
    "kw_hits": rng.poisson(0.15, n).clip(0, 3),
    "digit_ratio": rng.beta(1, 6, n).round(2),
    "unknown_handle": (rng.random(n) < 0.1).astype(int),
    "txns_last_hour": rng.poisson(1.0, n),
    "amount_ratio": rng.lognormal(0, 0.6, n).clip(0.1, 20).round(2),
    "device_changed": (rng.random(n) < 0.1).astype(int),
    "distance_km": rng.exponential(25, n).round(1),
})
z = (-5.0 + 0.00005*df.amount + 1.0*df.receiver_new + 1.6*df.receiver_flags + 1.4*df.kw_hits
     + 2.0*df.digit_ratio + 1.2*df.unknown_handle + 0.5*df.txns_last_hour + 0.10*df.amount_ratio
     + 1.5*df.device_changed + 0.01*df.distance_km + 1.5*(df.hour < 5))
df["is_fraud"] = ((z + rng.normal(0, 0.7, n)) > 0).astype(int)
df.to_csv("upi_transactions.csv", index=False)
print("Fraud rate: %.1f%%" % (df.is_fraud.mean()*100))

X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], df.is_fraud, test_size=0.2,
                                          stratify=df.is_fraud, random_state=42)
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
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
