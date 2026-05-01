import importlib.util
import os, sys, time, json, pickle
import numpy as np

if importlib.util.find_spec("tflite_runtime.interpreter") is not None:
    tflite = importlib.import_module("tflite_runtime.interpreter")
    Interpreter = tflite.Interpreter
    print("Using standalone tflite_runtime")
else:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generator import (gen_normal, gen_inner_race, gen_outer_race,
                             gen_ball_fault, CLASS_MAP, N_SAMPLES, FS)
from feature_extractor import extract_features

MODEL_DIR   = "models"
TFLITE_PATH = os.path.join(MODEL_DIR, "bearing_fault_model.tflite")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
RF_PATH     = os.path.join(MODEL_DIR, "random_forest.pkl")

CLASS_NAMES     = ["Normal", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]
EMOJIS          = ["✅", "🔴", "🟠", "🟡"]
ALARM_COUNT     = 3
N_LIVE_WINDOWS  = 20


class BearingEdgeInference:
    """
    Dual-path edge inference:
      Path A: Random Forest  — high-accuracy classification (98.3%)
      Path B: TFLite MLP     — measures edge latency (18 KB flatbuffer)

    On real hardware, Path A (RF via ONNX) or Path B (MLP) runs alone.
    Here both paths run in parallel to demonstrate the full system.
    """
    def __init__(self):
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        with open(RF_PATH, "rb") as f:
            self.rf = pickle.load(f)

        self.interp = Interpreter(model_path=TFLITE_PATH)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()
        self.out = self.interp.get_output_details()
        # warm-up
        dummy = np.zeros((1, self.inp[0]["shape"][1]), dtype=np.float32)
        self._tflite_run(dummy)

        kb = os.path.getsize(TFLITE_PATH) / 1024
        print(f"  RF loaded         ({os.path.getsize(RF_PATH)/1024:.0f} KB)")
        print(f"  TFLite loaded     ({kb:.1f} KB  — edge flatbuffer)")
        print(f"  Input shape       : {self.inp[0]['shape']}")

    def _tflite_run(self, x):
        self.interp.set_tensor(self.inp[0]["index"], x)
        self.interp.invoke()
        return self.interp.get_tensor(self.out[0]["index"])

    def predict(self, raw_seg):
        fd  = extract_features(raw_seg)
        fv  = np.array(list(fd.values()), dtype=np.float32).reshape(1, -1)
        fvs = self.scaler.transform(fv).astype(np.float32)

        # TFLite latency benchmark
        t0  = time.perf_counter()
        self._tflite_run(fvs)
        lat = (time.perf_counter() - t0) * 1000

        # RF classification (primary decision)
        probs = self.rf.predict_proba(fvs)[0]
        pred  = int(np.argmax(probs))
        conf  = float(probs[pred])
        return pred, conf, probs, lat


def stream_windows(name, gen_fn, engine, true_lbl, n=N_LIVE_WINDOWS):
    print(f"\n{'─'*62}")
    print(f"  SCENARIO: {name}   (true label: {CLASS_NAMES[true_lbl]})")
    print(f"{'─'*62}")
    print(f"  {'#':>3}  {'Predicted Class':<22} {'Conf':>6}  {'TFLite':>8}  Status")
    print(f"  {'─'*3}  {'─'*22} {'─'*6}  {'─'*8}  {'─'*10}")

    segs = gen_fn(n)
    lats, correct, alarm_streak, alarm_fired = [], 0, 0, False
    log = []

    for i, seg in enumerate(segs):
        pred, conf, probs, lat = engine.predict(seg)
        lats.append(lat)
        correct += int(pred == true_lbl)
        alarm_streak = alarm_streak + 1 if pred != 0 else 0
        alarm_now = (alarm_streak >= ALARM_COUNT) and not alarm_fired
        if alarm_now:
            alarm_fired = True
        status = "⚠️  ALARM!" if alarm_now else ("FAULT" if pred != 0 and pred != true_lbl else "")
        print(f"  {i+1:>3}  {EMOJIS[pred]} {CLASS_NAMES[pred]:<20} {conf:>6.1%}  {lat:>7.3f}ms  {status}")
        log.append({"window": i+1, "true": CLASS_NAMES[true_lbl],
                    "predicted": CLASS_NAMES[pred], "confidence": round(conf, 4),
                    "latency_ms": round(lat, 3), "correct": pred == true_lbl})

    acc = correct / n
    print(f"\n  Accuracy     : {acc:.1%}  ({correct}/{n} correct)")
    print(f"  Avg latency  : {np.mean(lats):.3f} ms  |  P95: {np.percentile(lats,95):.3f} ms")
    print(f"  Throughput   : {1000/np.mean(lats):.0f} windows/sec"
          f"  (window = {N_SAMPLES/FS*1000:.0f} ms of signal)")
    print(f"  Alarm fired  : {'YES ⚠️' if alarm_fired else 'NO'}")
    return log, acc, float(np.mean(lats))


def main():
    print("\n" + "═"*62)
    print("  EDGE DEPLOYMENT SIMULATOR — Bearing Fault Detection")
    print("═"*62)
    for p in [TFLITE_PATH, SCALER_PATH, RF_PATH]:
        if not os.path.exists(p):
            print(f"ERROR: Missing {p} — run train_model.py first.")
            sys.exit(1)

    print("\nInitialising edge inference engine …")
    engine = BearingEdgeInference()

    scenarios = [
        ("Healthy Motor",      gen_normal,      0),
        ("Inner Race Defect",  gen_inner_race,  1),
        ("Outer Race Defect",  gen_outer_race,  2),
        ("Ball Fault",         gen_ball_fault,  3),
    ]

    all_logs, accs, lats = {}, [], []
    for name, fn, lbl in scenarios:
        log, acc, lat = stream_windows(name, fn, engine, lbl)
        all_logs[name] = log
        accs.append(acc)
        lats.append(lat)

    print(f"\n{'═'*62}")
    print("  OVERALL EDGE DEPLOYMENT SUMMARY")
    print(f"{'═'*62}")
    print(f"  RF classification accuracy   : {np.mean(accs):.1%}")
    print(f"  TFLite inference latency     : {np.mean(lats):.3f} ms avg")
    print(f"  TFLite model size            : {os.path.getsize(TFLITE_PATH)/1024:.1f} KB")
    print(f"  Windows processed            : {N_LIVE_WINDOWS * len(scenarios)}")
    print(f"  Signal duration              : {N_LIVE_WINDOWS * len(scenarios) * N_SAMPLES / FS:.1f}s")
    print(f"\n  Deployable targets:")
    print(f"    • Raspberry Pi 4  (ARM Cortex-A72)")
    print(f"    • NVIDIA Jetson Nano")
    print(f"    • Arduino Portenta H7  (TFLite Micro)")
    print(f"    • STM32 + X-CUBE-AI")
    print(f"{'═'*62}\n")

    os.makedirs("reports", exist_ok=True)
    with open("reports/edge_deployment_log.json", "w") as f:
        json.dump({
            "metadata": {
                "tflite_kb": round(os.path.getsize(TFLITE_PATH)/1024, 1),
                "mean_latency_ms": round(np.mean(lats), 3),
                "overall_accuracy": round(np.mean(accs), 4),
            },
            "scenarios": all_logs,
        }, f, indent=2)
    print("  Log saved → reports/edge_deployment_log.json")


if __name__ == "__main__":
    main()
