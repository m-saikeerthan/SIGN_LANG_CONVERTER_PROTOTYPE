"""
Dynamic Video Processor for ISL Detection.
Processes continuous video recordings / clips:
1. Extracts frames from video.
2. Tracks hands across all frames.
3. Segments continuous gestures and eliminates redundant/duplicate frames.
4. Selects 2-3 representative keyframes per detected gesture.
5. Builds a structured dictionary mapping detected words to representative frames.
6. Synthesizes a grammatically correct sentence and Hindi translation.
"""

import os
import sys
import time
import tempfile
import base64
import cv2
import numpy as np

# Ensure src directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import (
    CONFIDENCE_THRESHOLD, DIFF_THRESHOLD,
    NUM_LANDMARKS, NUM_TWO_HAND_RAW
)
from hand_tracker import HandTracker
from gesture_classifier import GestureClassifier
from sentence_processor import SentenceProcessor


def encode_frame_to_b64(frame, quality=80, max_size=(480, 360)):
    """Resize and encode a BGR frame to base64 JPEG string."""
    h, w = frame.shape[:2]
    if w > max_size[0] or h > max_size[1]:
        scale = min(max_size[0] / w, max_size[1] / h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('utf-8')


class DynamicVideoProcessor:
    """
    Handles end-to-end processing of continuous video recordings for ISL.
    """

    def __init__(self):
        print("[ISL Video Processor] Initializing modules...")
        self.tracker = HandTracker(static_mode=True)
        self.classifier = GestureClassifier()
        self.sentence_processor = SentenceProcessor()

    def _load_video_frames(self, video_source):
        """
        Loads all frames from a video path or bytes.
        Returns list of (frame_idx, timestamp_sec, frame) and fps.
        """
        temp_file_path = None
        if isinstance(video_source, (bytes, bytearray)):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
            temp_file.write(video_source)
            temp_file.flush()
            temp_file.close()
            temp_file_path = temp_file.name
            cap_source = temp_file_path
        else:
            cap_source = video_source

        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise ValueError(f"Unable to open video source: {video_source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            timestamp = frame_idx / fps
            frames.append((frame_idx, timestamp, frame))
            frame_idx += 1

        cap.release()
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass

        return frames, fps

    def process_video(self, video_source):
        """
        Main pipeline for processing a video clip.

        Args:
            video_source: file path (str) or binary content (bytes).

        Returns:
            dict containing:
                - corrected_sentence: str
                - hindi_sentence: str
                - raw_words: list of str
                - gesture_dict: dict mapping word -> details (timestamps, confidence, 2-3 frames)
                - total_frames: int
                - processed_keyframes_count: int
                - execution_time_sec: float
        """
        start_time = time.time()
        frames, fps = self._load_video_frames(video_source)
        total_frames = len(frames)
        print(f"[ISL Video Processor] Loaded {total_frames} frames ({fps:.1f} FPS, {total_frames/fps:.2f}s).")

        if total_frames == 0:
            return {
                'corrected_sentence': "No video frames found.",
                'hindi_sentence': "",
                'raw_words': [],
                'gesture_dict': {},
                'total_frames': 0,
                'processed_keyframes_count': 0,
                'execution_time_sec': round(time.time() - start_time, 2)
            }

        # Step 1: Track hands and compute landmark displacement
        tracked_frames = []
        prev_landmarks = None

        for idx, timestamp, frame in frames:
            landmarks, raw_hands = self.tracker.process(frame)
            annotated = frame.copy()
            self.tracker.draw(annotated, raw_hands)

            if landmarks is not None:
                if prev_landmarks is not None:
                    motion = float(np.linalg.norm(landmarks - prev_landmarks))
                else:
                    motion = 0.0
                prev_landmarks = landmarks.copy()

                # Get raw classification probabilities
                from feature_engineer import compute_two_hand_features_single
                extended = compute_two_hand_features_single(landmarks)
                num_feats = len(self.classifier.mean)
                std_feats = (extended[:num_feats].reshape(1, num_feats) - self.classifier.mean) / self.classifier.std
                raw_probs = self.classifier._predict_raw(std_feats)

                sorted_p = np.sort(raw_probs)
                top1 = float(sorted_p[-1])
                top2 = float(sorted_p[-2]) if len(sorted_p) > 1 else 0.0
                pred_label = self.classifier.labels[np.argmax(raw_probs)]

                tracked_frames.append({
                    'idx': idx,
                    'timestamp': timestamp,
                    'frame': frame,
                    'annotated': annotated,
                    'landmarks': landmarks,
                    'motion': motion,
                    'pred_word': pred_label,
                    'confidence': top1,
                    'margin': top1 - top2,
                    'has_hand': True
                })
            else:
                prev_landmarks = None
                tracked_frames.append({
                    'idx': idx,
                    'timestamp': timestamp,
                    'frame': frame,
                    'annotated': annotated,
                    'landmarks': None,
                    'motion': 999.0,
                    'pred_word': None,
                    'confidence': 0.0,
                    'margin': 0.0,
                    'has_hand': False
                })

        # Step 2: Segment into gesture intervals
        # Group continuous hand-present sequences
        segments = []
        current_segment = []

        for item in tracked_frames:
            if item['has_hand'] and item['confidence'] >= CONFIDENCE_THRESHOLD and item['margin'] >= DIFF_THRESHOLD:
                current_segment.append(item)
            else:
                if len(current_segment) >= 3:  # minimum 3 frames for a valid sign gesture
                    segments.append(current_segment)
                current_segment = []

        if len(current_segment) >= 3:
            segments.append(current_segment)

        print(f"[ISL Video Processor] Found {len(segments)} candidate gesture segments.")

        # Step 3: Deduplicate frames within each segment and pick 2-3 representative keyframes
        gesture_results = []
        total_keyframes = 0

        for seg_idx, segment in enumerate(segments):
            # Check majority predicted word in this segment
            words_in_seg = [f['pred_word'] for f in segment]
            from collections import Counter
            counts = Counter(words_in_seg)
            majority_word, freq = counts.most_common(1)[0]
            
            # Filter segment frames to those matching the dominant gesture
            matched_frames = [f for f in segment if f['pred_word'] == majority_word]
            if len(matched_frames) < 2:
                continue

            # Sort matched frames by stability / confidence
            # Low motion + high confidence = best apex frames
            scored_frames = []
            for f in matched_frames:
                # Score: high confidence with low motion penalty
                score = f['confidence'] / (1.0 + f['motion'] * 2.0)
                scored_frames.append((score, f))

            scored_frames.sort(key=lambda x: x[0], reverse=True)

            # Smart deduplication: Pick 2-3 keyframes that are temporally spaced
            # (e.g. onset, apex/peak, offset)
            selected_keyframes = []
            
            # If segment has few frames (2 or 3), take all
            if len(matched_frames) <= 3:
                selected_keyframes = matched_frames
            else:
                # Select best apex frame
                best_frame = scored_frames[0][1]
                selected_keyframes.append(best_frame)

                # Find 1 earlier frame and 1 later frame with landmark difference to avoid exact duplicates
                early_candidates = [f for f in matched_frames if f['timestamp'] < best_frame['timestamp'] - 0.08]
                late_candidates = [f for f in matched_frames if f['timestamp'] > best_frame['timestamp'] + 0.08]

                if early_candidates:
                    # Pick highest score from early candidates
                    early_candidates.sort(key=lambda f: f['confidence'], reverse=True)
                    selected_keyframes.insert(0, early_candidates[0])

                if late_candidates:
                    # Pick highest score from late candidates
                    late_candidates.sort(key=lambda f: f['confidence'], reverse=True)
                    selected_keyframes.append(late_candidates[0])

                # If still only 1, pick second best scored with non-identical timestamp
                if len(selected_keyframes) < 2 and len(scored_frames) > 1:
                    selected_keyframes.append(scored_frames[1][1])

                # Limit to at most 3 keyframes
                selected_keyframes = selected_keyframes[:3]
                # Sort chronologically
                selected_keyframes.sort(key=lambda f: f['timestamp'])

            avg_conf = float(np.mean([f['confidence'] for f in selected_keyframes]))
            timestamps = [round(f['timestamp'], 2) for f in selected_keyframes]

            # Encode 2-3 keyframes to base64
            encoded_frames = [encode_frame_to_b64(f['annotated']) for f in selected_keyframes]
            total_keyframes += len(encoded_frames)

            gesture_results.append({
                'word': majority_word,
                'confidence': round(avg_conf, 3),
                'timestamps': timestamps,
                'frames': encoded_frames,
                'keyframe_indices': [f['idx'] for f in selected_keyframes]
            })

        # Step 4: Collapse consecutive identical words if separated by < 0.3s
        # and construct raw word list
        raw_words = []
        consolidated_dict = {}

        for g in gesture_results:
            word = g['word']
            if not raw_words or raw_words[-1] != word:
                raw_words.append(word)

            # Add to dictionary (accumulate frames if repeated)
            if word not in consolidated_dict:
                consolidated_dict[word] = {
                    'word': word,
                    'confidence': g['confidence'],
                    'timestamps': g['timestamps'],
                    'frames': g['frames'],
                    'count': 1
                }
            else:
                # Merge timestamps and append new frames up to 3 max
                consolidated_dict[word]['timestamps'].extend(g['timestamps'])
                existing_frames = consolidated_dict[word]['frames']
                for frame_b64 in g['frames']:
                    if len(existing_frames) < 3 and frame_b64 not in existing_frames:
                        existing_frames.append(frame_b64)
                consolidated_dict[word]['count'] += 1
                consolidated_dict[word]['confidence'] = max(consolidated_dict[word]['confidence'], g['confidence'])

        print(f"[ISL Video Processor] Detected sequence of words: {raw_words}")
        print(f"[ISL Video Processor] Extracted {total_keyframes} total representative keyframes for {len(consolidated_dict)} unique signs.")

        # Step 5: Sentence processing (NLP grammar correction + Hindi translation)
        if raw_words:
            nlp_result = self.sentence_processor.correct_and_translate(raw_words)
            corrected_sentence = nlp_result['english']
            hindi_sentence = nlp_result['hindi']
        else:
            corrected_sentence = "No clear sign language gestures detected in the video."
            hindi_sentence = ""

        exec_time = round(time.time() - start_time, 2)

        return {
            'corrected_sentence': corrected_sentence,
            'hindi_sentence': hindi_sentence,
            'raw_words': raw_words,
            'gesture_dict': consolidated_dict,
            'total_frames': total_frames,
            'processed_keyframes_count': total_keyframes,
            'execution_time_sec': exec_time
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process continuous ISL video input")
    parser.add_argument("--video", required=True, help="Path to input video file")
    args = parser.parse_args()

    processor = DynamicVideoProcessor()
    res = processor.process_video(args.video)

    print("\n" + "=" * 60)
    print("  DYNAMIC VIDEO PROCESSING RESULTS")
    print("=" * 60)
    print(f"Raw Signs Detected:  {res['raw_words']}")
    print(f"English Sentence:    {res['corrected_sentence']}")
    print(f"Hindi Translation:   {res['hindi_sentence']}")
    print(f"Dictionary Words:    {list(res['gesture_dict'].keys())}")
    for w, info in res['gesture_dict'].items():
        print(f"  [{w}]: Confidence {info['confidence']*100:.1f}%, Timestamps: {info['timestamps']}s, Keyframes: {len(info['frames'])}")
    print(f"Total Video Frames:  {res['total_frames']} -> Keyframes analyzed: {res['processed_keyframes_count']} in {res['execution_time_sec']}s")
    print("=" * 60)
