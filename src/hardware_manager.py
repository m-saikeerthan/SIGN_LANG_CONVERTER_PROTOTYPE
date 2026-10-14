"""
Hardware Manager & GPIO Controller for Wall-Mounted Raspberry Pi Appliance.
Controls optional status LEDs, buzzer, and shutdown button with graceful dummy fallback on PC.
"""

import sys
import time
import threading

# Default GPIO pin assignments for Raspberry Pi
PIN_LED_POWER = 17       # Green: System Ready / Power
PIN_LED_TRACKING = 27    # Yellow: Hand Present / Tracking
PIN_LED_CONFIRM = 22     # Blue: Gesture Confirmed
PIN_BUZZER = 23          # Active Buzzer (optional)
PIN_BTN_RESET = 24       # Physical Push Button (Clear / Reset)

HAS_RPIGPIO = False
try:
    import RPi.GPIO as GPIO
    HAS_RPIGPIO = True
except (ImportError, RuntimeError):
    HAS_RPIGPIO = False


class HardwareManager:
    """
    Manages physical indicators and hardware buttons.
    Operates safely in both embedded Linux environments and desktop PC development.
    """

    def __init__(self, enable_gpio=True):
        self.is_embedded = HAS_RPIGPIO and enable_gpio
        self._button_callback = None

        if self.is_embedded:
            try:
                GPIO.setwarnings(False)
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(PIN_LED_POWER, GPIO.OUT, initial=GPIO.LOW)
                GPIO.setup(PIN_LED_TRACKING, GPIO.OUT, initial=GPIO.LOW)
                GPIO.setup(PIN_LED_CONFIRM, GPIO.OUT, initial=GPIO.LOW)
                GPIO.setup(PIN_BUZZER, GPIO.OUT, initial=GPIO.LOW)
                
                # Setup physical reset button with pull-up resistor
                GPIO.setup(PIN_BTN_RESET, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                GPIO.add_event_detect(PIN_BTN_RESET, GPIO.FALLING, 
                                      callback=self._on_button_press, bouncetime=300)
                
                print(f"[Hardware] Initialized Raspberry Pi GPIO subsystems (LEDs: {PIN_LED_POWER}, {PIN_LED_TRACKING}, {PIN_LED_CONFIRM}).")
            except Exception as e:
                print(f"[Hardware] Warning: GPIO setup failed: {e}. Defaulting to software mock.")
                self.is_embedded = False
        else:
            # Running on PC / Development environment
            pass

    def _on_button_press(self, channel):
        if self._button_callback:
            self._button_callback()

    def register_reset_button(self, callback):
        """Register a callback function when the physical reset button is pressed."""
        self._button_callback = callback

    def set_power_led(self, state: bool):
        if self.is_embedded:
            GPIO.output(PIN_LED_POWER, GPIO.HIGH if state else GPIO.LOW)

    def set_tracking_led(self, state: bool):
        if self.is_embedded:
            GPIO.output(PIN_LED_TRACKING, GPIO.HIGH if state else GPIO.LOW)

    def set_confirm_led(self, state: bool):
        if self.is_embedded:
            GPIO.output(PIN_LED_CONFIRM, GPIO.HIGH if state else GPIO.LOW)

    def pulse_confirm(self, duration_sec=0.2):
        """Blink confirmation LED and pulse buzzer briefly."""
        if not self.is_embedded:
            return

        def _pulse():
            try:
                GPIO.output(PIN_LED_CONFIRM, GPIO.HIGH)
                GPIO.output(PIN_BUZZER, GPIO.HIGH)
                time.sleep(duration_sec)
                GPIO.output(PIN_LED_CONFIRM, GPIO.LOW)
                GPIO.output(PIN_BUZZER, GPIO.LOW)
            except Exception:
                pass

        threading.Thread(target=_pulse, daemon=True).start()

    def cleanup(self):
        """Clean up GPIO pins on shutdown."""
        if self.is_embedded:
            try:
                GPIO.output(PIN_LED_POWER, GPIO.LOW)
                GPIO.output(PIN_LED_TRACKING, GPIO.LOW)
                GPIO.output(PIN_LED_CONFIRM, GPIO.LOW)
                GPIO.cleanup()
                print("[Hardware] GPIO resources released.")
            except Exception:
                pass


# Global singleton instance
hardware = HardwareManager()
