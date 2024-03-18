#include <WiFi.h>
#include "esp_camera.h"
#include <HTTPClient.h>
#include "SR04.h"
#include <Servo.h>
#include <LiquidCrystal_I2C.h>
#include <Wire.h>

#define SERVO_PIN1 12
#define SERVO_PIN2 13

#define TRIG_PIN 14
#define ECHO_PIN 15

const char *ssid = "TrashCan";
const char *password = "123456789";
const char* host = "192.168.4.2";

Servo servo1;
Servo servo2;
SR04 sr04 = SR04(ECHO_PIN,TRIG_PIN);
LiquidCrystal_I2C lcd(0x27,16,2);

long distance;
bool receivedCategory = false;

void setup() {
  Serial.begin(9600);

  WiFi.mode(WIFI_AP);
  WiFi.softAP(ssid, password);

  Serial.println("Access Point started");
  Serial.print("IP Address: ");
  Serial.println(WiFi.softAPIP());

  servo1.attach(SERVO_PIN1);
  servo2.attach(SERVO_PIN2);
  lcd.init(); 
  lcd.clear();         
  lcd.backlight();
}

void loop() {
  distance = sr04.Distance();

  if(distance < 7) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      return;
    }

    String category = POSTRequest(fb); 
    esp_camera_fb_return(fb);

    displayCategory(category);

    if (garbageType == "trash" || garbageType == "Trash"){
      trash();

    } else if (garbageType == "metal can" || garbageType == "Metal can" || garbageType == "Metal Can"){
      metalCan();

    } else if (garbageType == "electronic" || garbageType == "Electronic"){
      electronic();
      
    } else if (garbageType == "compost" || garbageType == "Compost"){
      compost();
      
    } else if (garbageType == "plastic" || garbageType == "Plastic"){
      plastic();
      
    }

    delay(1000);

  }

}


void initializeCamera() {
camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_SVGA;
  config.jpeg_quality = 10;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
}

String POSTRequest(camera_fb_t rawImage) {
  WiFiClient client;
  if (client.connect(host, 8000)) {
    Serial.println("Connected to server");

    client.println("POST /upload/ HTTP/1.1");
    client.println("Host: " + String(host));
    client.println("Content-Type: image/jpeg");
    client.println("Connection: close");
    client.print("Content-Length: ");
    client.println(fb->len);
    client.println();
    client.write(fb->buf, fb->len);
    Serial.println("Image sent");

    while (client.connected()) {
      if (client.available()) {
        String line = client.readStringUntil('\r');      
        Serial.println(line);
      }
    }
  } else {
    Serial.println("Unable to connect to server");
  }

  client.stop();

  delay(5000);
}


void displayCategory(String category) {
  lcd.clear();         
  lcd.backlight(); 
  lcd.scrollDisplayRight();
  lcd.setCursor(0,0);
  lcd.print(category);
}


void trash() {
  Serial.println("Received trash");
  delay(1000);
  servo1.write(0);
  delay(2000);
  servo1.write(-180);
  servo2.write(50);
  delay(1000);
  
  servo.write(360);
  delay(1000);
  servo2.write(90);
}

void metalCan() {
  Serial.println("Received metal can");
  delay(1000);
  servo1.write(60);
  delay(2000);
  servo1.write(180);
  servo2.write(50);
  delay(1000);
  delay(1000);
  servo2.write(90);
}

void electronic() {
  Serial.println("Received electronic");
  delay(1000);
  servo1.write(90);
  delay(2000);
  servo1.write(180);
  servo2.write(50);
  delay(1000);
  delay(1000);
  servo2.write(90);
}

void compost() {
  Serial.println("Received compost");
  delay(1000);
  servo1.write(120);
  delay(2000);
  servo1.write(180);
  servo2.write(50);
  delay(1000);
  delay(1000);
  servo2.write(90);
}

void plastic() {
  Serial.println("Received plastic");
  delay(1000);
  servo2.write(50);
  delay(1000);
  delay(1000);
  servo2.write(90);
}

