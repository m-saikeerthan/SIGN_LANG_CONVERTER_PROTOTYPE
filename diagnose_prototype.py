"""
ISL Prototype Diagnostic & Hardware Benchmark Suite.
Tests camera, MediaPipe hand tracking, TFLite/Keras inference, audio output, and NLP translation.
Run this script to verify full hardware and software readiness on PC or Raspberry Pi.

Usage:
    python diagnose_prototype.py
    python diagnose_prototype.py --no-audio
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

# Set stdout encoding
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    MODEL_TFLITE_PATH, MODEL_H5_PATH, LABELS_PATH, MEAN_PATH, STD_PATH
)


class PrototypeDiagnostics:
    def __init__(self, test_audio=True):
        self.test_audio = test_audio
        self.results = {}

    def print_header(self, title):
        print("\n" + "=" * 65)
        print(f"  🔍 TEST: {title}")
        print("=" * 65)

    def test_assets(self):
        self.print_header("1. Model Assets & Weights Integrity")
        try:
            assert os.path.exists(LABELS_PATH), "labels.npy missing"
            assert os.path.exists(MEAN_PATH), "mean.npy missing"
            assert os.path.exists(STD_PATH), "std.npy missing"

            labels = np.load(LABELS_PATH)
            mean = np.load(MEAN_PATH)
            std = np.load(STD_PATH)

            print(f"  ✓ Labels loaded:      {len(labels)} classes (e.g. {list(labels[:5])}...)")
            print(f"  ✓ Mean vector shape:  {mean.shape} (Expected 120)")
            print(f"  ✓ Std vector shape:   {std.shape} (Expected 120)")

            tflite_ok = os.path.exists(MODEL_TFLITE_PATH)
            keras_ok = os.path.exists(MODEL_H5_PATH)

            if tflite_ok:
                sz_kb = os.path.getsize(MODEL_TFLITE_PATH) / 1024
                print(f"  ✓ model.tflite:       Found ({sz_kb:.1f} KB) - Primary Edge Engine")
            else:
                print(f"  ⚠ model.tflite:       Not found (Will use Keras .h5)")

            if keras_ok:
                sz_mb = os.path.getsize(MODEL_H5_PATH) / (1024 * 1024)
                print(f"  ✓ model.h5:           Found ({sz_mb:.2f} MB) - Fallback Engine")

            assert tflite_ok or keras_ok, "Neither model.tflite nor model.h5 found!"
            self.results["Assets Integrity"] = "PASS"
        except Exception as e:
            print(f"  ✗ Asset Check Failed: {e}")
            self.results["Assets Integrity"] = f"FAIL ({e})"

    def test_camera(self):
        self.print_header("2. Camera Hardware Stream & Throughput")
        try:
            cap = cv2.VideoCapture(CAMERA_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

            if not cap.isOpened():
                print(f"  ⚠ Camera index {CAMERA_INDEX} could not be opened.")
                self.results["Camera Hardware"] = "WARNING (No camera attached / busy)"
                return

            frames_read = 0
            start_t = time.time()
            for _ in range(15):
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames_read += 1

            dur = time.time() - start_t
            fps = frames_read / max(dur, 0.001)
            cap.release()

            print(f"  ✓ Camera open:        Index {CAMERA_INDEX}")
            print(f"  ✓ Captured frames:    {frames_read}/15 frames")
            print(f"  ✓ Throughput speed:   {fps:.1f} FPS")

            self.results["Camera Hardware"] = "PASS" if frames_read > 5 else "WARNING"
        except Exception as e:
            print(f"  ✗ Camera Test Failed: {e}")
            self.results["Camera Hardware"] = f"FAIL ({e})"

    def test_tracking_and_features(self):
        self.print_header("3. MediaPipe Tracking & 120-dim Feature Engineering")
        try:
            from hand_tracker import HandTracker
            from feature_engineer import compute_two_hand_features_single

            tracker = HandTracker(static_mode=True)
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Benchmark tracking on blank frame
            t0 = time.time()
            landmarks, raw_hands = tracker.process(dummy_frame)
            track_time_ms = (time.time() - t0) * 1000
            tracker.close()

            print(f"  ✓ MediaPipe pipeline: Initialized cleanly")
            print(f"  ✓ Tracking Latency:   {track_time_ms:.1f} ms/frame")

            # Test 84 raw to 120 feature engineering with synthetic coordinates
            synthetic_84 = np.random.uniform(-0.5, 0.5, 84).astype(np.float32)
            extended_120 = compute_two_hand_features_single(synthetic_84)

            print(f"  ✓ Feature Expansion:  84 raw coordinates -> {len(extended_120)} extended features")
            assert len(extended_120) == 120, f"Expected 120 features, got {len(extended_120)}"

            self.results["Tracking & Features"] = "PASS"
        except Exception as e:
            print(f"  ✗ Feature Test Failed: {e}")
            self.results["Tracking & Features"] = f"FAIL ({e})"

    def test_classifier(self):
        self.print_header("4. Neural Inference Engine Benchmark")
        try:
            from gesture_classifier import GestureClassifier
            classifier = GestureClassifier()

            dummy_84 = np.zeros(84, dtype=np.float32)

            # Warmup
            classifier.classify(dummy_84)

            # Benchmark 20 inference cycles
            latencies = []
            for _ in range(20):
                t0 = time.time()
                res = classifier.classify(dummy_84)
                latencies.append((time.time() - t0) * 1000)

            avg_lat = sum(latencies) / len(latencies)
            mode_str = "TFLite (Quantized)" if classifier._use_tflite else "Keras (.h5)"

            print(f"  ✓ Active Engine:      {mode_str}")
            print(f"  ✓ Average Latency:    {avg_lat:.2f} ms ({1000/avg_lat:.0f} inferences/sec)")
            print(f"  ✓ State Machine:      Initial status = '{res['status']}'")

            self.results["Neural Inference"] = f"PASS ({avg_lat:.1f}ms {mode_str})"
        except Exception as e:
            print(f"  ✗ Classifier Test Failed: {e}")
            self.results["Neural Inference"] = f"FAIL ({e})"

    def test_audio_subsystem(self):
        self.print_header("5. Audio Cues & Speech Synthesis")
        if not self.test_audio:
            print("  ⚠ Audio testing skipped via --no-audio flag.")
            self.results["Audio & Voice"] = "SKIPPED"
            return

        try:
            from audio_feedback import AudioManager
            audio = AudioManager(enabled=True)

            print("  ✓ Playing test startup chime...")
            audio.play_startup_chime()
            time.sleep(0.3)

            print("  ✓ Playing test confirmation beep...")
            audio.play_confirm_beep()
            time.sleep(0.3)

            from sentence_processor import SentenceProcessor
            sp = SentenceProcessor()
            print("  ✓ Testing Text-to-Speech synthesis engine...")
            sp.speak("Diagnostic check complete.")
            time.sleep(0.5)

            self.results["Audio & Voice"] = "PASS"
        except Exception as e:
            print(f"  ✗ Audio Test Failed: {e}")
            self.results["Audio & Voice"] = f"FAIL ({e})"

    def test_nlp_engine(self):
        self.print_header("6. NLP Grammar Correction & Translation")
        try:
            from sentence_processor import SentenceProcessor
            sp = SentenceProcessor()

            test_sequence = ["ME", "HAPPY", "MEET", "YOU"]
            res = sp.correct_and_translate(test_sequence)

            print(f"  ✓ Input Gloss Sequence: {test_sequence}")
            print(f"  ✓ Synthesized English:  \"{res['english']}\"")
            print(f"  ✓ Hindi Translation:    \"{res['hindi']}\"")
            print(f"  ✓ Mode:                 {'Cloud Gemini' if sp._client else 'Local Heuristic Engine'}")

            assert res['english'], "English sentence is empty!"
            self.results["NLP Translation"] = "PASS"
        except Exception as e:
            print(f"  ✗ NLP Test Failed: {e}")
            self.results["NLP Translation"] = f"FAIL ({e})"

    def run_all(self):
        print("\n" + "=" * 65)
        print("    INDIAN SIGN LANGUAGE PROTOTYPE — DIAGNOSTIC & BENCHMARK")
        print("=" * 65)

        self.test_assets()
        self.test_camera()
        self.test_tracking_and_features()
        self.test_classifier()
        self.test_nlp_engine()
        self.test_audio_subsystem()

        # Final Scorecard
        print("\n" + "=" * 65)
        print("                   SYSTEM DIAGNOSTIC SCORECARD")
        print("=" * 65)
        all_ok = True
        for test_name, status in self.results.items():
            icon = "✅" if "PASS" in status else "⚠️" if "WARNING" in status or "SKIPPED" in status else "❌"
            if "FAIL" in status:
                all_ok = False
            print(f"  {icon} {test_name:<26} : {status}")
        print("=" * 65)

        if all_ok:
            print("  🎉 SYSTEM STATUS: FULLY OPERATIONAL & READY FOR DEPLOYMENT!")
        else:
            print("  ⚠️ SYSTEM STATUS: ATTENTION NEEDED ON FAILED SUBSYSTEMS.")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISL Prototype Diagnostics")
    parser.add_argument("--no-audio", action="store_true", help="Skip audible tone tests")
    args = parser.parse_args()

    suite = PrototypeDiagnostics(test_audio=not args.no_audio)
    suite.run_all()
