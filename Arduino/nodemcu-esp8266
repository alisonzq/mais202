#include <ESP8266WiFi.h>
#include <WiFiClient.h>
#include "SR04.h"

const char* ssid = "";
const char* password = "";
const char* host = ""; 
const int port = 8000; 

const int trigPin = 9;
const int echoPin = 10;

SR04 sr04 = SR04(echoPin,trigPin);
long distance;

void setup() {
  Serial.begin(115200);
  pinMode(D0, OUTPUT);
  delay(10);

  Serial.println();
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  distance = sr04.Distance();
  if(distance < 7) {
    POSTRequest();
  }

}

void POSTRequest() {
    WiFiClient client;
  if (client.connect(host, port)) {
    Serial.println("Connected to server");

    client.print(String("POST /upload/") + " HTTP/1.1\r\n" +
                 "Host: " + host + "\r\n" +
                 "Connection: close\r\n" +
                 "\r\n"
                );

    while (client.connected()) {
      if (client.available()) {
        String line = client.readStringUntil('\r');
        digitalWrite(D0, HIGH);        
        Serial.println(line);
      }
    }
  } else {
    Serial.println("Unable to connect to server");
  }

  client.stop();

  delay(5000);
}
