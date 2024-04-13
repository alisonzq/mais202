#include <WiFi.h>
#include "esp_camera.h"
#include "SR04.h"
#include <ESP32Servo.h>
#include <Wire.h>
#include "base64.h"

#define CAMERA_MODEL_WROVER_KIT

#include "camera_pins.h"

#define SERVO_PIN1 12
#define SERVO_PIN2 14

#define TRIG_PIN 33
#define ECHO_PIN 32

#define LED_GREEN 13

const char *ssid = "trashcan";
const char *password = "12345678";
IPAddress local_IP(192,168,1,100);//Set the IP address of ESP32 itself
IPAddress gateway(192,168,1,10);   //Set the gateway of ESP32 itself
IPAddress subnet(255,255,255,0);  //Set the subnet mask for ESP32 itself
const char* host = "192.168.1.101";
camera_config_t config;

Servo servo1;
Servo servo2;
SR04 sr04 = SR04(ECHO_PIN,TRIG_PIN);

int pos1 = 0; 
int pos2 = 0;
long distance;
bool canThrowNewTrash = true;

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  initializeWifiAP();
  initializeCamera();

  pinMode(LED_GREEN, OUTPUT);

  digitalWrite(LED_GREEN, HIGH);

	ESP32PWM::allocateTimer(0);
	ESP32PWM::allocateTimer(1);
	ESP32PWM::allocateTimer(2);
	ESP32PWM::allocateTimer(3);
	servo1.setPeriodHertz(50);    // standard 50 hz servo
	servo1.attach(SERVO_PIN1, 1000, 2000);
  servo2.setPeriodHertz(50);    // standard 50 hz servo
	servo2.attach(SERVO_PIN2, 1000, 2000);
}

void loop() {
  distance = sr04.Distance();

  if(distance < 5 && canThrowNewTrash) {
    canThrowNewTrash = false;
    digitalWrite(LED_GREEN, LOW);
    String garbageType = "";
    camera_fb_t *fb;
    if (captureImage(&fb)) {
      esp_camera_fb_return(fb);
      garbageType = POSTRequest(fb);
    }

    if (garbageType == "\"trash\"" || garbageType == "Trash"){
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
    canThrowNewTrash = true;
    digitalWrite(LED_GREEN, HIGH);
    delay(1000);
  }


}

void initializeWifiAP() {
  Serial.println("Setting soft-AP configuration ... ");
  WiFi.disconnect();
  WiFi.mode(WIFI_AP);
  Serial.println(WiFi.softAPConfig(local_IP, gateway, subnet) ? "Ready" : "Failed!");
  Serial.println("Setting soft-AP ... ");
  boolean result = WiFi.softAP(ssid, password);
  if(result){
    Serial.println("Ready");
    Serial.println(String("Soft-AP IP address = ") + WiFi.softAPIP().toString());
  }else{
    Serial.println("Failed!");
  }
  Serial.println("Setup End");
}

bool captureImage(camera_fb_t **fb) {
  *fb = esp_camera_fb_get();
  if (!*fb) {
    Serial.println("Failed to capture image");
    return false;
  }
  Serial.println("image captured");
  return true;
}

String POSTRequest(camera_fb_t *fb) {
  WiFiClient client;
  if (client.connect(host, 8000)) {
    Serial.println("Connected to server");

    String imageData = base64::encode(fb->buf, fb->len);

    String postData = "{\"image\": \"" + imageData + "\"}";

    String postRequest = "POST /upload/ HTTP/1.1\r\n" +
                          String("Host: ") + String(host) + "\r\n" +
                          "Content-Type: application/json\r\n" +
                          "Connection: close\r\n" +
                          "Content-Length: " + String(postData.length()) + "\r\n\r\n" +
                          postData;
                          
    client.print(postRequest);

    client.flush();
    Serial.println("Image sent to server"); 

    bool headersPassed = false;

    while (client.connected()) {
      if (client.available()) {
        String response = client.readStringUntil('\n');
        if (!headersPassed) {
          if (response == "\r") {
            headersPassed = true;
          }
        } else {
          Serial.println("received garbage type: " + response);
          client.stop();
          return response;
        }
      }
    }
  } else {
    Serial.println("Unable to connect to server");
  }

  client.stop();
  return "";

}



void trash() {
  Serial.println("Received trash");
  delay(1000);
  rotateServo1(0);
  delay(2000);
  rotateServo1(-180);
  rotateServo2(50);
  delay(1000);
  
  rotateServo1(360);
  delay(1000);
  rotateServo2(90);
}

void metalCan() {
  Serial.println("Received metal can");
  delay(1000);
  rotateServo1(60);
  delay(2000);
  rotateServo1(180);
  rotateServo2(50);
  delay(1000);
  delay(1000);
  rotateServo2(90);
}

void electronic() {
  Serial.println("Received electronic");
  delay(1000);
  rotateServo1(90);
  delay(2000);
  rotateServo1(180);
  rotateServo2(50);
  delay(1000);
  delay(1000);
  rotateServo2(90);
}

void compost() {
  Serial.println("Received compost");
  delay(1000);
  rotateServo1(120);
  delay(2000);
  rotateServo1(180);
  rotateServo2(50);
  delay(1000);
  delay(1000);
  rotateServo2(90);
}

void plastic() {
  Serial.println("Received plastic");
  delay(1000);
  rotateServo2(50);
  delay(1000);
  delay(1000);
  rotateServo2(90);
}

void rotateServo1(int angle) {
  for (pos1 = 0; pos1 <= angle; pos1 += 1) {
		servo1.write(pos1);
		delay(1);
	}
}

void rotateServo2(int angle) {
  for (pos2 = 0; pos2 <= angle; pos2 += 1) {
		servo2.write(pos2);
		delay(1);
	}
}

void initializeCamera() {
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
  config.frame_size = FRAMESIZE_QVGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  //camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t * s = esp_camera_sensor_get();
  s->set_vflip(s, 0);        //1-Upside down, 0-No operation
  s->set_hmirror(s, 0);      //1-Reverse left and right, 0-No operation
  s->set_brightness(s, 1);   //up the brightness just a bit
  s->set_saturation(s, 1);  //lower the saturation

}
