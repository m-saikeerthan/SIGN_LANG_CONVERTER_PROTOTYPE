# Indian Sign Language (ISL) Recognition Wall-Mounted Prototype
## Comprehensive Technical Specification, Hardware Architecture, Bill of Materials, and Implementation Roadmap

---

## 1. Project Overview & Codebase Architecture

The Indian Sign Language (ISL) Converter is an end-to-end, multi-stage computer vision and natural language processing system. It translates continuous two-hand Indian Sign Language gestures captured from video input into grammatically sound English sentences and spoken audio output in real time.

```
+--------------------------------------------------------------------------------------------------+
|                                    ISL END-TO-END PIPELINE                                       |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|   +-----------------------+        +--------------------------+        +---------------------+   |
|   | 1. Video Frame Stream | -----> | 2. Hand Tracking Module  | -----> | 3. Feature Eng.     |   |
|   |    (640x480 @ 30 FPS) |        |    (MediaPipe Hands)     |        |    (84 -> 120 Feats)|   |
|   +-----------------------+        +--------------------------+        +---------------------+   |
|                                                                                   |              |
|                                                                                   v              |
|   +-----------------------+        +--------------------------+        +---------------------+   |
|   | 6. Audio Output Unit  | <----- | 5. Sentence Synthesis    | <----- | 4. Gesture Classifier|  |
|   |    (Speaker / TTS)    |        |    (Gemini LLM / Local)  |        |    (TFLite ResNet)  |   |
|   +-----------------------+        +--------------------------+        +---------------------+   |
|                                                                                                  |
+--------------------------------------------------------------------------------------------------+
```

### 1.1 Core Modules & Responsibilities

1. **`HandTracker` (`src/hand_tracker.py`):**
   - Implements Google MediaPipe Hands tracking for up to 2 hands concurrently (`MP_MAX_HANDS = 2`).
   - Extracts 21 3D landmarks ($x, y, z$) per hand.
   - Converts coordinates into wrist-centered relative coordinates ($x_i - x_{\text{wrist}}$, $y_i - y_{\text{wrist}}$) for scale and position invariance.
   - Sorts hands horizontally (left-to-right based on wrist $x$-coordinate) and concatenates them into an 84-dimensional vector ($21 \times 2 \times 2$). If only one hand is visible, the second hand is zero-padded.

2. **`FeatureEngineer` (`src/feature_engineer.py`):**
   - Expands the 84 raw coordinate features into a **120-dimensional extended feature vector**.
   - Computes:
     - **Fingertip-to-wrist Euclidean distances** for all 5 fingers.
     - **Inter-finger distances** (Thumb-Index, Index-Middle, Middle-Ring, Ring-Pinky).
     - **Fingertip angles** relative to the palm plane.
     - **Inter-hand Euclidean distances** (between left and right wrists and corresponding fingertips).
     - **Relative bounding box dimensions and aspect ratios**.

3. **`GestureClassifier` (`src/gesture_classifier.py`):**
   - Executes inference using the quantized **`model.tflite`** (115.9 KB) with fallback to Keras **`model.h5`** (1.4 MB).
   - Normalizes input vectors using precomputed dataset mean (`mean.npy`) and standard deviation (`std.npy`).
   - Applies an **Exponential Moving Average (EMA)** smoothing filter ($\alpha = 0.55$) across consecutive frame probability distributions:
     $$\text{EMA}_t = \alpha \cdot P_t + (1 - \alpha) \cdot \text{EMA}_{t-1}$$
   - Implements a confidence check ($\text{Confidence} \ge 80\%$) and difference margin threshold ($P_{\text{top1}} - P_{\text{top2}} \ge 0.20$).
   - Manages a **Gesture Hold State Machine** requiring the user to hold the sign steadily for **1.0 second** before confirmation to eliminate transitional noise.

4. **`SentenceProcessor` (`src/sentence_processor.py`):**
   - Indian Sign Language uses a **Subject-Object-Verb (SOV)** grammatical structure, whereas English uses **Subject-Verb-Object (SVO)**.
   - Converts collected word glosses (e.g., `["ME", "HAPPY", "MEET", "YOU"]`) into natural English (`"I am happy to meet you."`).
   - **Primary Engine:** Google Gemini LLM (`gemini-2.5-flash` / `gemini-1.5-flash`) with structured system instructions.
   - **Local Fallback Engine:** Rule-based SOV $\rightarrow$ SVO reordering, pronoun normalization (`_PRONOUN_MAP`), and verb extraction (`_VERB_HINTS`) with zero network latency.
   - Generates Hindi translations and triggers audio synthesis.

5. **`DynamicVideoProcessor` (`src/dynamic_video_processor.py`):**
   - Handles batch processing of recorded video streams, dynamic gesture segmentation, keyframe deduplication, and structured gesture-to-frame mapping.

---

## 2. Hardware Engineering & Feasibility Analysis

### 2.1 The "Arduino" Query: Why Microcontrollers Cannot Run This Directly

A standard Arduino board (such as the Uno, Mega, or Nano) or standard ESP32 cannot execute this machine learning pipeline independently:

| Requirement | MediaPipe + TFLite Pipeline | Standard Arduino (Uno / Mega) | ESP32-S3 (TinyML) | Single Board Computer (Raspberry Pi 4B / 5) |
| :--- | :--- | :--- | :--- | :--- |
| **CPU Architecture** | ARMv8 64-bit / x86_64 | 8-bit AVR RISC | 32-bit Xtensa Dual-Core | 64-bit Quad-Core ARM Cortex-A76/A72 @ 2.4GHz |
| **Clock Frequency** | $\ge 1.5 \text{ GHz}$ | $16 \text{ MHz}$ | $240 \text{ MHz}$ | $1.8 \text{ GHz} - 2.4 \text{ GHz}$ |
| **RAM (Memory)** | $\ge 1 \text{ GB}$ recommended | **2 KB (Uno) / 8 KB (Mega)** | **512 KB SRAM + 8 MB PSRAM** | **4 GB / 8 GB LPDDR4X** |
| **Operating System** | Linux / Windows / Android | None (Bare Metal) | FreeRTOS | Full 64-bit Linux (Debian/Raspberry Pi OS) |
| **MediaPipe Support**| Fully Supported | Unsupported | Unsupported | Fully Supported |
| **Real-time FPS** | 25 – 30 FPS | 0 FPS (Incompatible) | $<1$ FPS (Custom Tiny CNN only) | 25 – 32 FPS (Real-time) |

*MediaPipe Hands relies on BlazePalm (Single Shot Detector) and BlazeHand (3D regression mesh) deep networks requiring hundreds of MegaFLOPs per frame. An Arduino lacks the memory, instruction set, and floating-point registers needed.*

---

## 3. Recommended Hardware Architectures

### Architecture A: Standalone Edge AI Wall Appliance (Primary Recommendation)

A self-contained, wall-mounted unit where all image capture, hand tracking, neural classification, NLP sentence synthesis, and audio generation occur **on-device without needing any external computer or active internet connection for core sign detection**.

```
+---------------------------------------------------------------------------------------+
|                       WALL-MOUNTED ENCLOSURE (STANDALONE)                             |
|                                                                                       |
|   +-------------------+        +--------------------------------------------------+   |
|   | Wide-Angle Camera |        | Raspberry Pi 5 (4GB) / Pi 4B (4GB)               |   |
|   | 120° FOV / CSI    | -----> | - 64-bit Linux OS                                |   |
|   +-------------------+        | - MediaPipe Hands + TFLite ISL Model (115 KB)    |   |
|                                | - Local Rule-Based & Gemini NLP Engine           |   |
|                                | - Offline TTS Synthesis Engine                   |   |
|                                +-------------------------+------------------------+   |
|                                                          | I2S Digital Audio (GPIO)   |
|                                                          v                            |
|   +-------------------+        +--------------------------------------------------+   |
|   | 3W 4-Ohm Cavity   | <----- | MAX98357A I2S Class-D DAC & Amplifier Board      |   |
|   | Vocal Speaker     |        | (Decodes PCM digital audio to analog speaker)    |   |
|   +-------------------+        +--------------------------------------------------+   |
|                                                                                       |
+---------------------------------------------------------------------------------------+
                                           ^
                                           | 5V / 3A - 5A DC
+---------------------------------------------------------------------------------------+
|                    DIRECT 100V - 240V AC TO 5V DC POWER ADAPTER                       |
+---------------------------------------------------------------------------------------+
```

---

### Architecture B: Distributed Client-Server Node (Alternative)

If you specifically want a compact microcontroller mounted on the wall while your laptop or a home server handles the computation:

1. **Wall Node:** ESP32-S3-CAM + MAX98357A I2S Amplifier + Speaker.
2. **Network:** Local Wi-Fi (WebSocket or HTTP stream).
3. **Host Machine:** Your existing PC running `web/app.py` or a dedicated background Python server.
4. **Operation:** ESP32 captures and streams MJPEG video frames $\rightarrow$ Host PC runs MediaPipe, classifies gestures, formats the sentence, and sends synthesized PCM/MP3 audio back to the ESP32 to play over the speaker.

---

## 4. Comprehensive Bill of Materials (BOM) & Cost Estimation

### Standalone Edge AI Appliance (Option A)

| Component | Part / Model Specification | Purpose / Function | Quantity | Unit Cost (USD) | Unit Cost (INR) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Main Compute Board** | Raspberry Pi 5 (4GB RAM) or Raspberry Pi 4 Model B (4GB) | Runs MediaPipe, TFLite inference, NLP, and audio generation | 1 | $60.00 – $75.00 | ₹5,200 – ₹6,500 |
| **Camera Module** | Raspberry Pi Camera Module 3 (Wide 120° FOV) or 1080p USB Wide-Angle Webcam | Captures upper-body and two-hand gestures | 1 | $18.00 – $25.00 | ₹1,500 – ₹2,200 |
| **Audio DAC & Amplifier** | MAX98357A I2S 3.2W Class-D Mono Amplifier Module | Converts digital audio signals from GPIO into speaker drive | 1 | $4.50 – $7.00 | ₹380 – ₹600 |
| **Speaker Unit** | 3W 4Ω or 8Ω Enclosed Cavity Mini Speaker (40mm – 50mm) | Delivers speech output | 1 | $3.50 – $5.00 | ₹300 – ₹450 |
| **System Storage** | SanDisk Extreme / Ultra 64GB MicroSD Card (A2 / Class 10) | Holds OS, models, scripts, and logs | 1 | $9.00 – $12.00 | ₹750 – ₹1,000 |
| **Direct Power Supply** | 5V 5A (for Pi 5) or 5V 3A (for Pi 4B) USB-C AC/DC Wall Power Adapter | Supplies power directly from wall socket | 1 | $10.00 – $14.00 | ₹850 – ₹1,200 |
| **Thermal Management** | Active Aluminum Heatsink + 5V PWM Cooling Fan | Prevents thermal throttling during continuous operation | 1 | $5.00 – $8.00 | ₹400 – ₹700 |
| **Wall Enclosure & Bracket** | Custom 3D Printed PETG/ABS Enclosure or Acrylic Wall Box | Houses all boards, speaker grill, and camera mount | 1 | $8.00 – $15.00 | ₹650 – ₹1,250 |
| **Cabling & Hardware** | Female-to-Female Jumper Wires, M2.5/M3 Standoffs, Wall Screws | Interconnections and mechanical mounting | 1 Set | $4.00 – $6.00 | ₹350 – ₹500 |
| **Estimated Total** | | | | **$122.00 – $167.00** | **₹10,380 – ₹14,400** |

---

## 5. Wiring Diagram & Electrical Interconnections

### I2S Audio Module (MAX98357A) to Raspberry Pi GPIO Pinout

The MAX98357A I2S DAC amplifier connects to the Raspberry Pi 40-pin header as follows:

```
+------------------------+                     +------------------------+
|  MAX98357A I2S Board   |                     |   Raspberry Pi GPIO    |
+------------------------+                     +------------------------+
| LRC   (Left/Right Cloc)| <------------------ | GPIO 19 (Pin 35 - FS)  |
| BCLK  (Bit Clock)      | <------------------ | GPIO 18 (Pin 12 - CLK) |
| DIN   (Digital Data In)| <------------------ | GPIO 21 (Pin 40 - DOUT)|
| GND   (Ground)         | <------------------ | GND     (Pin 6 / 9 /14)|
| VIN   (Power Input)    | <------------------ | 5V      (Pin 2 / 4)    |
| GAIN  (Gain Select)    | ------------------> | Tied to GND (Default)  |
| SD    (Shutdown)       | ------------------> | Unconnected (On)       |
| SPK+  / SPK-           | ------------------> | Connect to 3W Speaker  |
+------------------------+                     +------------------------+
```

---

## 6. Physical Wall Installation, Optical FOV & Power Engineering

```
                          WALL SURFACE
                               │
                ┌──────────────┴──────────────┐
                │   [ Speaker Sound Grill ]   │
                │                             │
                │   📷 Camera Lens (120° FOV) │  <--- 1.40 m - 1.50 m (4.6 - 5.0 ft)
                │      (Tilted 12° Downward)  │
                │                             │
                │   🔘 Audio Status Indicator │
                │   [ Internal Fan Exhaust ]  │
                └──────────────┬──────────────┘
                               │
                               │ 2-Core Insulated DC Power Cable
                               ▼
                ┌─────────────────────────────┐
                │  5V / 3A-5A Wall AC Adapter │ <--- 100V - 240V AC Wall Outlet
                └─────────────────────────────┘
```

1. **Mounting Height:** Install the device at **1.45 meters (4.75 feet)** above the floor.
2. **Camera Field of View & Tilt:** Use a **120° wide-angle lens** tilted **$10^\circ \text{ to } 15^\circ$ downward**. This geometry captures the user from the waist to the top of the head when standing **1.0 to 1.8 meters** away, ensuring two-hand gestures remain in the frame.
3. **Direct Power Configuration:**
   - Connect the 5V DC power adapter directly to the wall mains outlet.
   - For completely concealed installations, an AC-to-DC internal converter module (e.g., Mean Well IRM-20-5 or Hi-Link HLK-PM01 5V) can be embedded inside a standard electrical conduit box behind the mount.
4. **Enclosure Ventilation:** The enclosure should feature intake vents on the lower edge and exhaust vents on the upper edge to facilitate natural thermal convection, assisted by an internal 5V micro-fan.

---

## 7. Headless Software Architecture for Screenless Voice-Only Operation

For a dedicated wall unit with **no screen and audio-only output**, the detection pipeline is run in headless mode. The display windows (`cv2.imshow`) are omitted, and auditory feedback signals are added.

### Headless Engine Script (`src/headless_prototype.py`)

Here is the operational script designed for the embedded prototype:

```python
"""
Headless Standalone ISL Detection & Voice Output Engine for Wall Prototype.
Runs continuously on boot, captures video, classifies gestures, formats sentences,
and delivers voice output over the attached speaker.
"""

import sys
import os
import time
import cv2
import numpy as np

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, NO_HAND_TIMEOUT,
    CONFIDENCE_THRESHOLD, DIFF_THRESHOLD, GESTURE_HOLD_TIME
)
from hand_tracker import HandTracker
from gesture_classifier import GestureClassifier
from sentence_processor import SentenceProcessor


class HeadlessISLApp:
    def __init__(self):
        print("[ISL Prototype] Initializing Hardware Subsystems...")
        self.tracker = HandTracker(static_mode=False)
        self.classifier = GestureClassifier()
        self.sentence_processor = SentenceProcessor()

        # State management
        self.sentence = []
        self.last_confirmed = ""
        self._no_hand_since = None
        self._holding_sound_played = False

        # Open video capture device
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print("[ISL Prototype] FATAL: Camera device could not be opened.")
            sys.exit(1)

        print("[ISL Prototype] System Ready. Awaiting Gestures...")

    def _play_tone(self, frequency=800, duration=0.08):
        """Optional subtle audio cue for headless user feedback."""
        pass  # Can generate small tone via numpy / sounddevice / I2S

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                # Process hand tracking
                landmarks, _ = self.tracker.process(frame)

                if landmarks is not None:
                    self._no_hand_since = None
                    result = self.classifier.classify(landmarks)

                    if result['status'] == 'confirmed':
                        word = result['word']
                        if not self.sentence or self.sentence[-1] != word:
                            self.sentence.append(word)
                            print(f"[ISL Prototype] Added word: {word} -> Current Sequence: {self.sentence}")
                            self._play_tone(frequency=1000, duration=0.1)

                else:
                    # Hands lowered / no hand detected
                    if self._no_hand_since is None:
                        self._no_hand_since = time.time()
                    elif time.time() - self._no_hand_since >= NO_HAND_TIMEOUT:
                        if self.sentence:
                            print(f"[ISL Prototype] Finalizing Sentence: {self.sentence}")
                            # Convert ISL gloss sequence to English sentence and speak aloud
                            res = self.sentence_processor.process(self.sentence)
                            print(f"[ISL Prototype] Spoken: \"{res['english']}\"")
                            self.sentence = []
                            self.last_confirmed = ""
                            self.classifier.reset()
                        self._no_hand_since = None

                time.sleep(0.005)

        except KeyboardInterrupt:
            print("[ISL Prototype] Shutting down...")
        finally:
            self.cap.release()
            self.tracker.close()


if __name__ == "__main__":
    app = HeadlessISLApp()
    app.run()
```

---

## 8. Automatic Startup on Power-On (`systemd` Service)

To make the wall unit operate as an appliance that begins functioning as soon as power is connected, configure a Linux `systemd` service:

1. Create a service file: `/etc/systemd/system/isl-detector.service`
   ```ini
   [Unit]
   Description=ISL Sign Language Recognition Wall Unit
   After=network.target sound.target

   [Service]
   Type=simple
   User=pi
   WorkingDirectory=/home/pi/sign-language-detection
   Environment="PYTHONUNBUFFERED=1"
   ExecStart=/usr/bin/python3 /home/pi/sign-language-detection/src/headless_prototype.py
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

2. Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable isl-detector.service
   sudo systemctl start isl-detector.service
   ```

---

## 9. Implementation Roadmap & Project Timeline

```
+--------------------------------------------------------------------------------------------------+
|                                    14-DAY BUILD ROADMAP                                          |
+--------------------------------------------------------------------------------------------------+
| Days 1 - 3   | Component Procurement (Pi 5/4B, Camera, MAX98357A, Speaker, Power Supply)         |
| Days 4 - 5   | OS Setup, Audio Driver (I2S) Verification & Python Environment Configuration      |
| Days 6 - 8   | Deploying MediaPipe, TFLite Model & Running On-Device Benchmark Tests              |
| Days 9 - 10  | Integrating Offline TTS & Headless State Machine Logic                            |
| Days 11 - 12 | 3D Enclosure Fabrication, Electrical Wiring & Wall Mounting Assembly              |
| Days 13 - 14 | Real-World Gesture Accuracy Calibration & Lighting Field Testing                  |
+--------------------------------------------------------------------------------------------------+
```

---

## 10. Summary & Recommended Action Plan

1. **Hardware Acquisition:** Procure a **Raspberry Pi 5 (4GB)** or **Raspberry Pi 4 Model B (4GB)**, **Raspberry Pi Camera Module 3 (Wide-Angle)**, **MAX98357A I2S DAC Module**, and a **3W 4Ω cavity speaker**.
2. **Audio Setup:** Connect the MAX98357A I2S DAC to the Raspberry Pi GPIO pins (18, 19, 21, 5V, GND) and enable `dtoverlay=hifiberry-dac` in `/boot/firmware/config.txt`.
3. **Codebase Deployment:** Clone the repository, copy the pre-trained weights (`model.tflite`, `labels.npy`, `mean.npy`, `std.npy`), and run `headless_prototype.py`.
4. **Physical Assembly:** Install inside an enclosure mounted at a height of 4.5–5.0 ft, connect the 5V power supply, and activate the `systemd` daemon.
