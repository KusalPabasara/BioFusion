"""
BioFusion Kiosk — Serial Bridge
Communicates with the ESP32 controller over USB serial.
Falls back gracefully if ESP32 is not connected (development mode).
"""

import serial
import serial.tools.list_ports
import threading
import time
import logging

logger = logging.getLogger(__name__)


class SerialBridge:
    """Manages serial communication with the ESP32 kiosk controller."""

    def __init__(self, port=None, baud_rate=115200, timeout=2):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.connection = None
        self.connected = False
        self._lock = threading.Lock()

    def connect(self):
        """Attempt to connect to the ESP32. Returns True if successful."""
        # Try specified port first
        if self.port:
            if self._try_connect(self.port):
                return True

        # Auto-detect ESP32
        detected = self._auto_detect()
        if detected:
            if self._try_connect(detected):
                return True

        logger.warning("ESP32 not found — running in software-only mode")
        self.connected = False
        return False

    def _try_connect(self, port):
        """Try connecting to a specific serial port."""
        try:
            self.connection = serial.Serial(
                port=port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            # Wait for ESP32 to reset after connection
            time.sleep(2)
            # Flush any startup messages
            self.connection.reset_input_buffer()
            # Test connection
            if self._ping():
                self.connected = True
                self.port = port
                logger.info(f"Connected to ESP32 on {port}")
                return True
            else:
                self.connection.close()
                return False
        except (serial.SerialException, OSError) as e:
            logger.warning(f"Failed to connect on {port}: {e}")
            return False

    def _auto_detect(self):
        """Auto-detect ESP32 serial port."""
        ports = serial.tools.list_ports.comports()
        for p in ports:
            desc = (p.description or "").lower()
            mfr = (p.manufacturer or "").lower()
            # Common ESP32 identifiers
            if any(kw in desc for kw in ["cp210", "ch340", "ch9102", "usb serial", "uart", "arduino", "mega"]):
                logger.info(f"Auto-detected possible Arduino/ESP32: {p.device} ({p.description})")
                return p.device
            if any(kw in mfr for kw in ["silicon labs", "wch", "espressif", "arduino"]):
                logger.info(f"Auto-detected possible Arduino/ESP32: {p.device} ({p.manufacturer})")
                return p.device
        return None

    def _ping(self):
        """Send a PING to verify ESP32 is responding."""
        try:
            response = self._send_raw("PING")
            return response == "PONG"
        except Exception:
            return False

    def _send_raw(self, command):
        """Send a raw command and read the response."""
        if not self.connection or not self.connection.is_open:
            return None
        with self._lock:
            self.connection.write(f"{command}\n".encode())
            response = self.connection.readline().decode().strip()
            return response

    def _send(self, command):
        """Send a command to the ESP32. No-op if not connected."""
        if not self.connected:
            logger.debug(f"[SW-ONLY] Would send: {command}")
            return True
        try:
            response = self._send_raw(command)
            if response == "ACK":
                logger.debug(f"Sent: {command} → ACK")
                return True
            else:
                logger.warning(f"Sent: {command} → unexpected: {response}")
                return False
        except Exception as e:
            logger.error(f"Serial error sending {command}: {e}")
            self.connected = False
            return False

    # ─── Public API ──────────────────────────────────────────────────────

    def lights_on(self):
        """Turn on the NeoPixel LED strip."""
        return self._send("LIGHTS_ON")

    def lights_off(self):
        """Turn off the NeoPixel LED strip."""
        return self._send("LIGHTS_OFF")

    def set_status(self, status):
        """
        Set the status LED state.
        Valid states: 'idle', 'scan', 'process', 'normal', 'pneumonia'
        """
        cmd = f"STATUS_{status.upper()}"
        return self._send(cmd)

    def disconnect(self):
        """Close the serial connection."""
        if self.connection and self.connection.is_open:
            self.connection.close()
            logger.info("Serial connection closed")
        self.connected = False

    def get_status_info(self):
        """Get connection status info for the UI."""
        return {
            "connected": self.connected,
            "port": self.port if self.connected else None,
            "mode": "hardware" if self.connected else "software-only"
        }
