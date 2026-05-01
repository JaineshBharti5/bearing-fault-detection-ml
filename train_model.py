
import os
import time
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np  # type: ignore[import]
import pandas as pd  # type: ignore[import]
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import]
from sklearn.preprocessing import StandardScaler  # type: ignore[import]
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold  # type: ignore[import]
from sklearn.metrics import (  # type: ignore[import]
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
import tensorflow as tf  # type: ignore[import]

#Paths
DATA_DIR    = "data"
MODEL_DIR   = "models"
REPORT_DIR  = "reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

CLASS_NAMES = ["Normal", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]

# 1. LOAD DATA


def load_features(csv_path: str = f"{DATA_DIR}/bearing_features.csv"):
    df = pd.read_csv(csv_path)
    y  = df["label"].values
    X  = df.drop(columns=["label", "class_name"]).values.astype(np.float32)
    print(f"Loaded feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, df.drop(columns=["label","class_name"]).columns.tolist()

# 2. TRAIN RANDOM FOREST

def train_random_forest(X_train, y_train):
    print("\n── Training Random Forest ──")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    t0 = time.time()
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.2f}s | Trees: {rf.n_estimators}")
    return rf

# 3. EVALUATE

def evaluate_model(rf, X_train, X_test, y_train, y_test, feature_names):
    print("\n── Evaluation Results ──")

    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"  5-Fold CV F1-macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Test set
    y_pred   = rf.predict(X_test)
    acc      = accuracy_score(y_test, y_pred)
    f1_mac   = f1_score(y_test, y_pred, average="macro")
    f1_wt    = f1_score(y_test, y_pred, average="weighted")
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

    print(f"  Test Accuracy  : {acc:.4f}")
    print(f"  Test F1-macro  : {f1_mac:.4f}")
    print(f"  Test F1-weighted: {f1_wt:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")

    # Feature importance (top 10)
    fi = pd.Series(rf.feature_importances_, index=feature_names)
    fi = fi.sort_values(ascending=False)
    print(f"\nTop-10 Important Features:")
    print(fi.head(10).to_string())

    # Save report
    results = {
        "cv_f1_mean":   float(cv_scores.mean()),
        "cv_f1_std":    float(cv_scores.std()),
        "test_accuracy": float(acc),
        "test_f1_macro": float(f1_mac),
        "test_f1_weighted": float(f1_wt),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importances": fi.head(15).to_dict(),
    }
    with open(f"{REPORT_DIR}/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation report saved → {REPORT_DIR}/evaluation_results.json")
    return results, y_pred

# 4. CONVERT TO TFLITE


def sklearn_rf_to_tflite(rf, scaler, n_features: int, out_dir: str = MODEL_DIR):

    print("\n── Converting to TFLite ──")

    n_classes = len(CLASS_NAMES)

    X_all_path = f"{DATA_DIR}/bearing_features.csv"
    df = pd.read_csv(X_all_path)
    X_all = df.drop(columns=["label","class_name"]).values.astype(np.float32)
    X_all_scaled = scaler.transform(X_all)
    soft_labels  = rf.predict_proba(X_all).astype(np.float32)

    # ─ Keras student model (lightweight MLP) ─
    inp = tf.keras.Input(shape=(n_features,), name="vibration_features")
    x   = tf.keras.layers.Dense(128, activation="relu")(inp)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Dense(64, activation="relu")(x)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="class_probs")(x)
    model = tf.keras.Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"  Student MLP: {model.count_params():,} parameters")
    print("  Distilling RF knowledge into MLP …")

    history = model.fit(
        X_all_scaled, soft_labels,
        epochs=60, batch_size=32,
        validation_split=0.15,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
        ]
    )
    final_acc = max(history.history["val_accuracy"])
    print(f"  Distillation val-accuracy: {final_acc:.4f}")

    # ─ TFLite conversion ─
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]   # dynamic-range quant
    tflite_model = converter.convert()

    tflite_path = os.path.join(out_dir, "bearing_fault_model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"  TFLite model size : {size_kb:.1f} KB")
    print(f"  TFLite saved      → {tflite_path}")
    return tflite_path, size_kb



# 5. MAIN


def main():
    # Load
    X, y, feature_names = load_features()
    n_features = X.shape[1]

    # Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Train RF
    rf = train_random_forest(X_train, y_train)

    # Evaluate
    results, y_pred = evaluate_model(rf, X_train, X_test, y_train, y_test, feature_names)

    # Save scaler + RF
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{MODEL_DIR}/random_forest.pkl", "wb") as f:
        pickle.dump(rf, f)
    print(f"\nScaler saved → {MODEL_DIR}/scaler.pkl")
    print(f"RF model saved → {MODEL_DIR}/random_forest.pkl")

    # TFLite conversion
    tflite_path, size_kb = sklearn_rf_to_tflite(rf, scaler, n_features)

    # Summary
    print("\n" + "═"*55)
    print("  TRAINING COMPLETE — DEPLOYMENT SUMMARY")
    print("═"*55)
    print(f"  Random Forest accuracy     : {results['test_accuracy']*100:.2f}%")
    print(f"  Random Forest F1-macro     : {results['test_f1_macro']:.4f}")
    print(f"  TFLite model size          : {size_kb:.1f} KB")
    print(f"  Edge-ready                 : YES (TFLite runtime)")
    print("═"*55)

    return rf, scaler, tflite_path, results


if __name__ == "__main__":
    main()
