const int relayPin = 7;
const unsigned long relayPulseMs = 300;

bool relayActive = false;
unsigned long relayStartTime = 0;

void relayOff() {
  digitalWrite(relayPin, HIGH);  // Active LOW: HIGH = OFF
  relayActive = false;
}

void setup() {
  digitalWrite(relayPin, HIGH);
  pinMode(relayPin, OUTPUT);

  Serial.begin(9600);
}

void loop() {
  while (Serial.available() > 0) {
    char command = Serial.read();

    if (command == 'R') {
      digitalWrite(relayPin, LOW);  // Relay ON
      relayStartTime = millis();
      relayActive = true;

      Serial.println("REJECT: RELAY ON");
    }
    else if (command == 'X') {
      relayOff();

      Serial.println("RELAY OFF");
    }
  }

  // Turn relay OFF automatically after 300 ms
  if (relayActive && millis() - relayStartTime >= relayPulseMs) {
    relayOff();

    Serial.println("RELAY PULSE FINISHED");
  }
}
