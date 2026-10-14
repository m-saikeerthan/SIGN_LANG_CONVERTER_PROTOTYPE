"""
Acoustic Audio Feedback & Tone Synthesizer for Screenless ISL Prototypes.
Provides instant audio cues for headless / wall-mounted operation without external sound files.
Supports Windows, macOS, and Linux/Raspberry Pi (I2S DAC / ALSA / PulseAudio).
"""

import os
import sys
import math
import wave
import struct
import tempfile
import threading
import subprocess

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class AudioManager:
    """
    Synthesizes and plays low-latency tone cues and voice output.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._temp_dir = tempfile.gettempdir()

    def play_tone(self, frequency=800, duration_sec=0.1, volume=0.5):
        """Play a pure sine wave tone at the specified frequency and duration."""
        if not self.enabled:
            return

        def _play():
            with self._lock:
                try:
                    if HAS_WINSOUND and sys.platform.startswith('win'):
                        freq = max(37, min(32767, int(frequency)))
                        dur_ms = max(10, int(duration_sec * 1000))
                        winsound.Beep(freq, dur_ms)
                    else:
                        # Linux / macOS: Generate in-memory WAV buffer
                        sample_rate = 22050
                        num_samples = int(sample_rate * duration_sec)
                        wav_path = os.path.join(self._temp_dir, f"isl_tone_{int(frequency)}_{int(duration_sec*1000)}.wav")

                        if not os.path.exists(wav_path):
                            with wave.open(wav_path, 'wb') as wav_file:
                                wav_file.setnchannels(1)  # Mono
                                wav_file.setsampwidth(2)  # 16-bit
                                wav_file.setframerate(sample_rate)
                                
                                # Sine wave samples with soft attack/decay envelope to prevent clicking
                                frames = bytearray()
                                for i in range(num_samples):
                                    t = i / sample_rate
                                    # Envelope
                                    env = min(i / (0.01 * sample_rate + 1), (num_samples - i) / (0.01 * sample_rate + 1), 1.0)
                                    val = int(volume * env * 32767.0 * math.sin(2.0 * math.pi * frequency * t))
                                    frames.extend(struct.pack('<h', val))
                                wav_file.writeframes(frames)

                        # Play via ALSA (aplay / paplay) on Linux / Raspberry Pi
                        if sys.platform.startswith('linux'):
                            for player in ['aplay -q', 'paplay', 'play -q']:
                                cmd = f"{player} {wav_path} > /dev/null 2>&1"
                                if os.system(cmd) == 0:
                                    break
                        elif sys.platform == 'darwin':
                            os.system(f"afplay {wav_path} > /dev/null 2>&1")
                except Exception as e:
                    # Non-fatal if audio device is busy
                    pass

        t = threading.Thread(target=_play, daemon=True)
        t.start()

    def play_startup_chime(self):
        """Ascending three-tone chime signifying hardware & ML ready."""
        if not self.enabled:
            return
        def _chime():
            for freq in [523, 659, 784]:  # C5 -> E5 -> G5
                self.play_tone(frequency=freq, duration_sec=0.08, volume=0.4)
                import time
                time.sleep(0.09)
        threading.Thread(target=_chime, daemon=True).start()

    def play_hand_detected(self):
        """Subtle high tick when hand enters optical field."""
        self.play_tone(frequency=1400, duration_sec=0.03, volume=0.25)

    def play_hold_tick(self, progress=0.5):
        """Hold progress tick with rising pitch as hold completes."""
        pitch = 700 + int(progress * 500)
        self.play_tone(frequency=pitch, duration_sec=0.03, volume=0.3)

    def play_confirm_beep(self):
        """High affirmative beep when gesture is locked & confirmed."""
        self.play_tone(frequency=1046, duration_sec=0.12, volume=0.5)  # C6

    def play_sentence_finalizing(self):
        """Two-tone chime when sentence is sent to NLP engine."""
        def _double():
            self.play_tone(frequency=880, duration_sec=0.06, volume=0.4)
            import time
            time.sleep(0.07)
            self.play_tone(frequency=1175, duration_sec=0.10, volume=0.5)
        threading.Thread(target=_double, daemon=True).start()

    def play_error_buzz(self):
        """Low warning buzz on hardware error or camera failure."""
        self.play_tone(frequency=260, duration_sec=0.3, volume=0.6)


# Global singleton instance
audio = AudioManager()


if __name__ == "__main__":
    print("[Audio Feedback Test] Testing chime sounds...")
    import time
    audio.play_startup_chime()
    time.sleep(0.5)
    audio.play_confirm_beep()
    time.sleep(0.5)
    audio.play_sentence_finalizing()
    time.sleep(1.0)
    print("[Audio Feedback Test] Complete.")
