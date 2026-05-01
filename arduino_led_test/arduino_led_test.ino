// Pin Definitions for 4 LEDs
const int PIN_NEUTRAL = 2;
const int PIN_STRESSED = 3;
const int PIN_FOCUSED = 4;
const int PIN_DISTRACTED = 5;

void setup() {
  Serial.begin(9600); // Must match Python's baud rate
  pinMode(PIN_NEUTRAL, OUTPUT);
  pinMode(PIN_STRESSED, OUTPUT);
  pinMode(PIN_FOCUSED, OUTPUT);
  pinMode(PIN_DISTRACTED, OUTPUT);
  
  // Flash all LEDs once to confirm hardware is working
  digitalWrite(PIN_NEUTRAL, HIGH); digitalWrite(PIN_STRESSED, HIGH);
  digitalWrite(PIN_FOCUSED, HIGH); digitalWrite(PIN_DISTRACTED, HIGH);
  delay(500);
  allOff();
}

void loop() {
  if (Serial.available() > 0) {
    char stateChar = Serial.read();
    allOff(); // Reset before lighting the new state

    switch (stateChar) {
      case 'N': digitalWrite(PIN_NEUTRAL, HIGH); break;
      case 'S': digitalWrite(PIN_STRESSED, HIGH); break;
      case 'F': digitalWrite(PIN_FOCUSED, HIGH); break;
      case 'D': digitalWrite(PIN_DISTRACTED, HIGH); break;
    }
  }
}

void allOff() {
  digitalWrite(PIN_NEUTRAL, LOW);
  digitalWrite(PIN_STRESSED, LOW);
  digitalWrite(PIN_FOCUSED, LOW);
  digitalWrite(PIN_DISTRACTED, LOW);
}