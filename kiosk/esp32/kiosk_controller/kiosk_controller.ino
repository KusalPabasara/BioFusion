/*
 * BioFusion Kiosk — ESP32 Controller
 * 
 * Controls:
 *   - WS2812 NeoPixel LED strip (scan lighting)
 *   - Onboard/external RGB LED (status indicator)
 * 
 * Serial protocol (115200 baud, newline-terminated):
 *   Host → ESP32:  LIGHTS_ON | LIGHTS_OFF | STATUS_IDLE | STATUS_SCAN |
 *                   STATUS_PROCESS | STATUS_NORMAL | STATUS_PNEUMONIA
 *   ESP32 → Host:  ACK
 * 
 * Wiring:
 *   - NeoPixel Data → GPIO 5  (with 330Ω resistor in series)
 *   - Status LED R   → GPIO 25
 *   - Status LED G   → GPIO 26
 *   - Status LED B   → GPIO 27
 *   - NeoPixel 5V    → External 5V supply (not ESP32 3.3V)
 *   - Common GND     → Shared between ESP32, NeoPixel, LED
 */

#include <Adafruit_NeoPixel.h>

// ─── Pin Definitions ────────────────────────────────────────────────────────
#define NEOPIXEL_PIN    5
#define NEOPIXEL_COUNT  30   // Adjust to your strip length

#define STATUS_LED_R    25
#define STATUS_LED_G    26
#define STATUS_LED_B    27

// ─── NeoPixel Setup ─────────────────────────────────────────────────────────
Adafruit_NeoPixel strip(NEOPIXEL_COUNT, NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);

// ─── State ──────────────────────────────────────────────────────────────────
String inputBuffer = "";
bool lightsOn = false;
unsigned long lastPulse = 0;
int pulseValue = 0;
int pulseDirection = 1;

// ─── Status Colors (R, G, B) ────────────────────────────────────────────────
// Matching BioFusion palette: no red
enum KioskStatus {
  STATUS_IDLE,       // Sapphire Blue pulse (#2563EB)
  STATUS_SCAN,       // White solid
  STATUS_PROCESS,    // Amber (#F59E0B)
  STATUS_NORMAL,     // Emerald (#10B981)
  STATUS_PNEUMONIA   // Amber (#F59E0B) — NOT red (design decision)
};

KioskStatus currentStatus = STATUS_IDLE;

void setup() {
  Serial.begin(115200);
  
  // Status LED pins
  pinMode(STATUS_LED_R, OUTPUT);
  pinMode(STATUS_LED_G, OUTPUT);
  pinMode(STATUS_LED_B, OUTPUT);
  
  // Initialize NeoPixel strip
  strip.begin();
  strip.setBrightness(200);  // 0-255
  strip.show();              // All off
  
  setStatusIdle();
  
  Serial.println("BioFusion Kiosk Controller Ready");
}

void loop() {
  // ── Read serial commands ──
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else if (c != '\r') {
      inputBuffer += c;
    }
  }
  
  // ── Animate status LED ──
  updateStatusAnimation();
}

// ─── Command Processing ─────────────────────────────────────────────────────
void processCommand(String cmd) {
  cmd.trim();
  
  if (cmd == "LIGHTS_ON") {
    lightsOn = true;
    setStripWhite();
    Serial.println("ACK");
    
  } else if (cmd == "LIGHTS_OFF") {
    lightsOn = false;
    setStripOff();
    Serial.println("ACK");
    
  } else if (cmd == "STATUS_IDLE") {
    currentStatus = STATUS_IDLE;
    Serial.println("ACK");
    
  } else if (cmd == "STATUS_SCAN") {
    currentStatus = STATUS_SCAN;
    Serial.println("ACK");
    
  } else if (cmd == "STATUS_PROCESS") {
    currentStatus = STATUS_PROCESS;
    Serial.println("ACK");
    
  } else if (cmd == "STATUS_NORMAL") {
    currentStatus = STATUS_NORMAL;
    Serial.println("ACK");
    
  } else if (cmd == "STATUS_PNEUMONIA") {
    currentStatus = STATUS_PNEUMONIA;
    Serial.println("ACK");
    
  } else if (cmd == "PING") {
    Serial.println("PONG");
    
  } else {
    Serial.println("ERR:UNKNOWN_CMD");
  }
}

// ─── NeoPixel Strip Control ─────────────────────────────────────────────────
void setStripWhite() {
  for (int i = 0; i < NEOPIXEL_COUNT; i++) {
    strip.setPixelColor(i, strip.Color(255, 255, 255));
  }
  strip.show();
}

void setStripOff() {
  strip.clear();
  strip.show();
}

// ─── Status LED Control ─────────────────────────────────────────────────────
void setStatusColor(int r, int g, int b) {
  analogWrite(STATUS_LED_R, r);
  analogWrite(STATUS_LED_G, g);
  analogWrite(STATUS_LED_B, b);
}

void setStatusIdle() {
  // Sapphire Blue
  setStatusColor(37, 99, 235);
}

void updateStatusAnimation() {
  unsigned long now = millis();
  
  switch (currentStatus) {
    case STATUS_IDLE:
      // Pulsing blue
      if (now - lastPulse > 15) {
        lastPulse = now;
        pulseValue += pulseDirection * 3;
        if (pulseValue >= 235) { pulseValue = 235; pulseDirection = -1; }
        if (pulseValue <= 30)  { pulseValue = 30;  pulseDirection = 1; }
        setStatusColor(pulseValue * 37 / 235, pulseValue * 99 / 235, pulseValue);
      }
      break;
      
    case STATUS_SCAN:
      // Solid white
      setStatusColor(255, 255, 255);
      break;
      
    case STATUS_PROCESS:
      // Pulsing amber
      if (now - lastPulse > 20) {
        lastPulse = now;
        pulseValue += pulseDirection * 4;
        if (pulseValue >= 245) { pulseValue = 245; pulseDirection = -1; }
        if (pulseValue <= 50)  { pulseValue = 50;  pulseDirection = 1; }
        setStatusColor(pulseValue, pulseValue * 158 / 245, pulseValue * 11 / 245);
      }
      break;
      
    case STATUS_NORMAL:
      // Solid emerald green
      setStatusColor(16, 185, 129);
      break;
      
    case STATUS_PNEUMONIA:
      // Solid amber (NOT red — design decision)
      setStatusColor(245, 158, 11);
      break;
  }
}
