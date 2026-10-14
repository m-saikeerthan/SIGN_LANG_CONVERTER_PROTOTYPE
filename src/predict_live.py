"""
Fast live ISL gesture prediction script using two-hand tracking and TFLite inference.
"""

import os
import sys
import time
import cv2
import numpy as np

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    COLOR_GREEN, COLOR_YELLOW, COLOR_RED, COLOR_WHITE, COLOR_DARK_BG,
    FONT, FONT_SMALL, FONT_MEDIUM, FONT_LARGE, FONT_THICKNESS
)
from hand_tracker import HandTracker
from gesture_classifier import GestureClassifier
from sentence_processor import SentenceProcessor


def run_live():
    print("=" * 55)
    print("  ISL Live Prediction (1 or 2 Hands)")
    print("  Controls: 'c' = Clear sentence | 'q' = Quit")
    print("=" * 55)

    tracker = HandTracker(static_mode=False)
    classifier = GestureClassifier()
    sentence_proc = SentenceProcessor()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print(f"[ISL] Error: Cannot open camera index {CAMERA_INDEX}")
        return

    sentence = []
    last_added_word = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for natural selfie mirror
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        landmarks, raw_hands = tracker.process(frame)
        tracker.draw(frame, raw_hands)

        word_text = "—"
        conf_text = "0%"
        status_text = "No Hand"
        status_color = COLOR_YELLOW

        if landmarks is not None:
            result = classifier.classify(landmarks)
            status = result['status']
            conf = result['confidence']

            if result['word']:
                word_text = result['word']
                conf_text = f"{int(conf * 100)}%"

            if status == 'confirmed':
                status_text = "CONFIRMED"
                status_color = COLOR_GREEN
                word = result['word']
                if not sentence or sentence[-1] != word:
                    sentence.append(word)
                    last_added_word = word
            elif status == 'holding':
                status_text = f"Holding ({int(result['hold_progress'] * 100)}%)"
                status_color = (0, 200, 255)
            else:
                status_text = "Detecting..."
                status_color = COLOR_WHITE
        else:
            classifier.reset()

        # UI Overlay
        cv2.rectangle(frame, (0, 0), (w, 60), COLOR_DARK_BG, -1)
        cv2.putText(frame, f"Word: {word_text} ({conf_text})", (15, 30),
                    FONT, FONT_MEDIUM, status_color, FONT_THICKNESS)
        cv2.putText(frame, f"Status: {status_text}", (15, 52),
                    FONT, FONT_SMALL, COLOR_WHITE, 1)

        # Sentence bar
        cv2.rectangle(frame, (0, h - 45), (w, h), (20, 20, 20), -1)
        sentence_str = " ".join(sentence) if sentence else "Sentence will appear here..."
        cv2.putText(frame, sentence_str, (15, h - 15),
                    FONT, FONT_MEDIUM, COLOR_WHITE, 1)

        cv2.imshow("ISL Live Predictor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if sentence:
                # Speak finalized sentence on manual clear
                nlp_res = sentence_proc.correct_and_translate(sentence)
                print(f"[ISL] Spoken: {nlp_res['english']}")
                sentence_proc.speak(nlp_res['english'])
            sentence = []
            classifier.reset()
        elif key == ord('q'):
            break

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live()