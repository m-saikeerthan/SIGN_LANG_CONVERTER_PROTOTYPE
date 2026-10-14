"""
Headless Standalone ISL Detection & Voice Output Engine for Wall Prototype.
Runs continuously on boot without GUI display windows.
Captures camera video, tracks hands, classifies gestures, formats sentences,
and delivers voice speech output and acoustic feedback over the connected speaker.
"""

import sys
import os
import time
import signal
import argparse
import datetime
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
from audio_feedback import AudioManager
from hardware_manager import HardwareManager


def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class HeadlessISLPredictor:
    """Headless prototype processor for wall-mounted sign language device."""

    def __init__(self, camera_idx=CAMERA_INDEX, hold_time=GESTURE_HOLD_TIME, 
                 confidence=CONFIDENCE_THRESHOLD, enable_cues=True):
        print(f"[{get_timestamp()}] [ISL Prototype] Initializing Hardware & ML Subsystems...")

        self.audio = AudioManager(enabled=enable_cues)
        self.hardware = HardwareManager(enable_gpio=True)

        # 1. Initialize vision & classifier modules
        self.tracker = HandTracker(static_mode=False)
        self.classifier = GestureClassifier()
        self.sentence_processor = SentenceProcessor()

        # 2. State variables
        self.sentence = []
        self.last_confirmed = ""
        self._no_hand_since = None
        self._is_running = True
        self.hold_time = hold_time
        self.confidence_thresh = confidence

        # Register physical reset button callback
        self.hardware.register_reset_button(self.clear_sentence)

        # 3. Initialize Camera
        self.cap = cv2.VideoCapture(camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print(f"[{get_timestamp()}] [ISL Prototype] FATAL: Camera device ({camera_idx}) could not be opened!")
            self.audio.play_error_buzz()
            sys.exit(1)

        # Set Power / Ready LED
        self.hardware.set_power_led(True)

        # Play system startup chime
        self.audio.play_startup_chime()

        print(f"[{get_timestamp()}] [ISL Prototype] Ready! {len(self.classifier.labels)} gestures loaded.")
        print(f"[{get_timestamp()}] [ISL Prototype] System running headlessly. Awaiting signs...")

    def clear_sentence(self):
        """Reset current sentence sequence (triggered by button or timeout)."""
        print(f"[{get_timestamp()}] [ISL Prototype] Resetting sentence state.")
        self.sentence = []
        self.last_confirmed = ""
        self.classifier.reset()
        self.hardware.set_tracking_led(False)

    def stop(self, *args):
        """Signal handler for graceful termination."""
        print(f"\n[{get_timestamp()}] [ISL Prototype] Shutdown signal received. Exiting...")
        self._is_running = False

    def run(self):
        """Continuous main loop for camera processing and voice synthesis."""
        # Handle systemd / terminal termination signals
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

        try:
            while self._is_running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                # Flip for mirror orientation
                frame = cv2.flip(frame, 1)

                # Process hand tracking
                landmarks, _ = self.tracker.process(frame)

                if landmarks is not None:
                    self.hardware.set_tracking_led(True)
                    self._no_hand_since = None
                    result = self.classifier.classify(landmarks)

                    if result['status'] == 'confirmed':
                        word = result['word']
                        if not self.sentence or self.sentence[-1] != word:
                            self.sentence.append(word)
                            print(f"[{get_timestamp()}] [ISL Prototype] Confirmed Word: [{word}] | Sequence: {' '.join(self.sentence)}")
                            
                            # Audible and visual confirmation cue
                            self.audio.play_confirm_beep()
                            self.hardware.pulse_confirm()

                else:
                    self.hardware.set_tracking_led(False)
                    # No hands detected in frame
                    if self._no_hand_since is None:
                        self._no_hand_since = time.time()
                    elif time.time() - self._no_hand_since >= NO_HAND_TIMEOUT:
                        if self.sentence:
                            print(f"\n[{get_timestamp()}] [ISL Prototype] Finalizing Sentence: {' '.join(self.sentence)}")
                            self.audio.play_sentence_finalizing()

                            # Process grammar & speak out via speaker
                            result = self.sentence_processor.process(self.sentence)
                            print(f"[{get_timestamp()}] [ISL Prototype] English Spoken: \"{result['english']}\"")
                            if result.get('hindi'):
                                print(f"[{get_timestamp()}] [ISL Prototype] Hindi Meaning:  \"{result['hindi']}\"")

                            # Clear state for next sentence
                            self.sentence = []
                            self.last_confirmed = ""
                            self.classifier.reset()

                        self._no_hand_since = None

                # Yield briefly to keep CPU usage optimal
                time.sleep(0.005)

        finally:
            self._cleanup()

    def _cleanup(self):
        """Release camera, audio, and hardware resources."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'tracker'):
            self.tracker.close()
        if hasattr(self, 'hardware'):
            self.hardware.cleanup()
        print(f"[{get_timestamp()}] [ISL Prototype] Hardware cleanly shut down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless Standalone ISL Wall Appliance")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera device index")
    parser.add_argument("--hold-time", type=float, default=GESTURE_HOLD_TIME, help="Hold duration for confirmation (seconds)")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--no-cues", action="store_true", help="Disable acoustic tone cues")
    args = parser.parse_args()

    app = HeadlessISLPredictor(
        camera_idx=args.camera,
        hold_time=args.hold_time,
        confidence=args.confidence,
        enable_cues=not args.no_cues
    )
    app.run()
