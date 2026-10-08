"""
Headless Standalone ISL Detection & Voice Output Engine for Wall Prototype.
Runs continuously on boot without GUI display windows.
Captures camera video, tracks hands, classifies gestures, formats sentences,
and delivers voice speech output over the connected speaker.
"""

import sys
import os
import time
import cv2
import numpy as np

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, NO_HAND_TIMEOUT,
    CONFIDENCE_THRESHOLD, DIFF_THRESHOLD, GESTURE_HOLD_TIME
)
from hand_tracker import HandTracker
from gesture_classifier import GestureClassifier
from sentence_processor import SentenceProcessor


class HeadlessISLPredictor:
    """Headless prototype processor for wall-mounted sign language device."""

    def __init__(self):
        print("[ISL Prototype] Initializing Hardware Subsystems...")

        # 1. Initialize vision & classifier modules
        self.tracker = HandTracker(static_mode=False)
        self.classifier = GestureClassifier()
        self.sentence_processor = SentenceProcessor()

        # 2. State variables
        self.sentence = []
        self.last_confirmed = ""
        self._no_hand_since = None
        self._fps_samples = []
        self._prev_time = time.time()

        # 3. Initialize Camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print("[ISL Prototype] FATAL: Camera device could not be opened!")
            sys.exit(1)

        print(f"[ISL Prototype] Ready! {len(self.classifier.labels)} gestures loaded.")
        print("[ISL Prototype] System is running headlessly (Voice Output Only).")

    def run(self):
        """Continuous main loop for camera processing and voice synthesis."""
        print("[ISL Prototype] Listening for hand gestures...")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                # Flip for mirror orientation
                frame = cv2.flip(frame, 1)

                # Process hand tracking
                landmarks, _ = self.tracker.process(frame)

                if landmarks is not None:
                    self._no_hand_since = None
                    result = self.classifier.classify(landmarks)

                    if result['status'] == 'confirmed':
                        word = result['word']
                        if not self.sentence or self.sentence[-1] != word:
                            self.sentence.append(word)
                            print(f"[ISL Prototype] Confirmed Word: [{word}] | Current Sentence: {' '.join(self.sentence)}")

                else:
                    # No hands detected in frame
                    if self._no_hand_since is None:
                        self._no_hand_since = time.time()
                    elif time.time() - self._no_hand_since >= NO_HAND_TIMEOUT:
                        if self.sentence:
                            print(f"\n[ISL Prototype] Finalizing Sentence: {' '.join(self.sentence)}")
                            # Process grammar & speak out via speaker
                            result = self.sentence_processor.process(self.sentence)
                            print(f"[ISL Prototype] English Spoken: \"{result['english']}\"")
                            if result.get('hindi'):
                                print(f"[ISL Prototype] Hindi Meaning:  \"{result['hindi']}\"")

                            # Clear state for next sentence
                            self.sentence = []
                            self.last_confirmed = ""
                            self.classifier.reset()

                        self._no_hand_since = None

                # Small yield to prevent 100% single-core spin
                time.sleep(0.005)

        except KeyboardInterrupt:
            print("\n[ISL Prototype] Stopping system...")
        finally:
            self._cleanup()

    def _cleanup(self):
        """Release camera and tracker resources."""
        self.cap.release()
        self.tracker.close()
        print("[ISL Prototype] Hardware cleanly released.")


if __name__ == "__main__":
    app = HeadlessISLPredictor()
    app.run()
