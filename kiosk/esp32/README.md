# ESP32 Kiosk Controller — Wiring Guide

## Components (BOM)

| Component | Qty | Notes |
|---|---|---|
| ESP32 DevKit V1 | 1 | Any ESP32 board with USB |
| WS2812B NeoPixel LED Strip | 1 | 30 LEDs (adjust `NEOPIXEL_COUNT` in sketch) |
| RGB Common-Cathode LED | 1 | For status indicator |
| 330Ω Resistor | 1 | NeoPixel data line |
| 220Ω Resistors | 3 | RGB LED current limiting |
| 1000µF Capacitor | 1 | NeoPixel power smoothing |
| 5V Power Supply | 1 | For NeoPixel strip (≥2A for 30 LEDs) |
| USB Cable | 1 | ESP32 to laptop |
| Breadboard + Jumper Wires | — | For prototyping |

## Pin Assignments

| ESP32 Pin | Connection |
|---|---|
| GPIO 5 | NeoPixel Data (via 330Ω resistor) |
| GPIO 25 | Status LED — Red (via 220Ω) |
| GPIO 26 | Status LED — Green (via 220Ω) |
| GPIO 27 | Status LED — Blue (via 220Ω) |
| GND | Common ground (LED, NeoPixel, PSU) |

## Wiring Diagram

```
                     ESP32 DevKit
                    ┌────────────┐
                    │            │
              GPIO5 ├──[330Ω]──→ NeoPixel Data In
                    │            │
             GPIO25 ├──[220Ω]──→ RGB LED (R)
             GPIO26 ├──[220Ω]──→ RGB LED (G)
             GPIO27 ├──[220Ω]──→ RGB LED (B)
                    │            │
                GND ├────────────┤── NeoPixel GND ── PSU GND ── RGB LED (-)
                    │            │
                USB ├──── To Laptop (serial + power)
                    └────────────┘

    5V PSU ──→ NeoPixel 5V
    5V PSU ──→ [1000µF cap] ──→ GND
```

## Software Setup

1. Install [Arduino IDE](https://www.arduino.cc/en/software) or [PlatformIO](https://platformio.org/)
2. Add ESP32 board support:
   - Arduino IDE: File → Preferences → Board URLs → add `https://dl.espressif.com/dl/package_esp32_index.json`
   - Install "ESP32 by Espressif" from Board Manager
3. Install library: **Adafruit NeoPixel** (Library Manager)
4. Select board: **ESP32 Dev Module**
5. Upload `kiosk_controller.ino`

## Serial Protocol

Baud rate: **115200**, newline-terminated (`\n`)

| Direction | Command | Description |
|---|---|---|
| Host → ESP32 | `LIGHTS_ON` | Turn on NeoPixel strip (white) |
| Host → ESP32 | `LIGHTS_OFF` | Turn off NeoPixel strip |
| Host → ESP32 | `STATUS_IDLE` | Blue pulsing status LED |
| Host → ESP32 | `STATUS_SCAN` | White solid status LED |
| Host → ESP32 | `STATUS_PROCESS` | Amber pulsing status LED |
| Host → ESP32 | `STATUS_NORMAL` | Green solid status LED |
| Host → ESP32 | `STATUS_PNEUMONIA` | Amber solid status LED |
| Host → ESP32 | `PING` | Connection test |
| ESP32 → Host | `ACK` | Command acknowledged |
| ESP32 → Host | `PONG` | Ping response |

## Status LED Colors

| State | Color | Hex |
|---|---|---|
| Idle | Sapphire Blue (pulsing) | `#2563EB` |
| Scanning | White (solid) | `#FFFFFF` |
| Processing | Amber (pulsing) | `#F59E0B` |
| Normal Result | Emerald (solid) | `#10B981` |
| Pneumonia Result | Amber (solid) | `#F59E0B` |

> **Note**: No red is used anywhere — this is a deliberate BioFusion design decision.
