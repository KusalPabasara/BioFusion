/*
 * BioFusion Kiosk — ESP32 Controller
 * 
 * Controls:
 *   - Simple White LED strip (scan lighting) via MOSFET/Relay
 * 
 * Serial protocol (115200 baud, newline-terminated):
 *   Host → ESP32:  LIGHTS_ON | LIGHTS_OFF | STATUS_IDLE | STATUS_SCAN |
 *                   STATUS_PROCESS | STATUS_NORMAL | STATUS_PNEUMONIA
 *   ESP32 → Host:  ACK
 */

// ─── Pin Definitions ────────────────────────────────────────────────────────
#define LED_STRIP_PIN    5 // Change this if you connected your MOSFET/Relay to a different pin

// ─── State ──────────────────────────────────────────────────────────────────
String inputBuffer = "";

void setup() {
  Serial.begin(115200);
  
  // Setup LED strip pin as output and turn it off by default
  pinMode(LED_STRIP_PIN, OUTPUT);
  digitalWrite(LED_STRIP_PIN, LOW); 
  
  Serial.println("BioFusion Kiosk Controller Ready (Simple White LED)");
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
}

// ─── Command Processing ─────────────────────────────────────────────────────
void processCommand(String cmd) {
  cmd.trim();
  
  if (cmd == "LIGHTS_ON") {
    digitalWrite(LED_STRIP_PIN, HIGH);
    Serial.println("ACK");
    
  } else if (cmd == "LIGHTS_OFF") {
    digitalWrite(LED_STRIP_PIN, LOW);
    Serial.println("ACK");
    
  } else if (cmd.startsWith("STATUS_")) {
    // We ignore status commands since we don't have a status LED anymore, 
    // but we still acknowledge them so the Python backend doesn't time out.
    Serial.println("ACK");
    
  } else if (cmd == "PING") {
    Serial.println("PONG");
    
  } else {
    Serial.println("ERR:UNKNOWN_CMD");
  }
}
