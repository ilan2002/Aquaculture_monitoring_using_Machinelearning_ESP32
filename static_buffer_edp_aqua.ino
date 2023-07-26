/* Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

/* Includes ---------------------------------------------------------------- */
#include <aquaculture_inferencing.h>

#include <OneWire.h>
#include <DallasTemperature.h>

#include <WiFi.h>
#include <PubSubClient.h>
#include <LiquidCrystal_I2C.h>
LiquidCrystal_I2C lcd(0x3F,16,2);

#define SENSOR_PIN 17 // ESP32 pin GIOP17 connected to DS18B20 sensor's DATA pin
OneWire oneWire(SENSOR_PIN);
DallasTemperature DS18B20(&oneWire);

const char* ssid = "ilan";
const char* password = "7094837175";
const char* mqtt_server = " 192.168.223.80";
WiFiClient espClient;
PubSubClient client(espClient);

float features[3], temp, ph;;
float Value=0;
const int potPin=A0;
int count = 0, flag = 0;


/**
 * @brief      Copy raw feature data in out_ptr
 *             Function called by inference library
 *
 * @param[in]  offset   The offset
 * @param[in]  length   The length
 * @param      out_ptr  The out pointer
 *
 * @return     0
 */
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

void print_inference_result(ei_impulse_result_t result);

/**
 * @brief      Arduino setup function
 */
void setup()
{
    // put your setup code here, to run once:
    Serial.begin(115200);
    // comment out the below line to cancel the wait for USB connection (needed for native USB)
    while (!Serial);
    DS18B20.begin();  
    pinMode(potPin,INPUT);
    Serial.println("AQUACULTURE MONITORING");
    lcd.init();         
    lcd.backlight();
    WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");

  client.setServer(mqtt_server, 1883);
  while (!client.connected()) {
    if (client.connect("ESP32_client")) {
      Serial.println("Connected to MQTT broker");
    } else {
      Serial.print("Failed to connect to MQTT broker, rc=");
      Serial.print(client.state());
      Serial.println(" retrying in 5 seconds");
      delay(5000);
    }
  }
    delay(1000);
}

/**
 * @brief      Arduino main function
 */
void loop()
{
    count++;
    flag = 0;  

    DS18B20.requestTemperatures();       // send the command to get temperatures
    temp = DS18B20.getTempCByIndex(0); 
    Value= analogRead(potPin);
    //Serial.print(Value);
    //Serial.print(" | ");
    float voltage=Value*(3.3/4095.0);
    ph=(3.3*voltage);
    Serial.println(ph);
    Serial.println(temp);
    static char phsensor[7];
    dtostrf(ph, 6, 2, phsensor);
    static char tempsensor[7];
    dtostrf(temp, 6, 2, tempsensor);
    client.publish("ph_sensor", phsensor);
    client.publish("temp_sensor", tempsensor);

    features[0] = ph;

    features[1] = temp;

    features[2] = 5.52;

    ei_printf("inferencing esp32 to machine learing model (ESP32)\n");

    if (sizeof(features) / sizeof(float) != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        ei_printf("The size of your 'features' array is not correct. Expected %lu items, but had %lu\n",
            EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, sizeof(features) / sizeof(float));
        delay(1000);
        return;
    }

    ei_impulse_result_t result = { 0 };

    // the features are stored into flash, and we don't want to load everything into RAM
    signal_t features_signal;
    features_signal.total_length = sizeof(features) / sizeof(features[0]);
    features_signal.get_data = &raw_feature_get_data;

    // invoke the impulse
    EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false /* debug */);
    if (res != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", res);
        return;
    }

    // print inference return code
    ei_printf("run_classifier returned: %d\r\n", res);
    print_inference_result(result);

    delay(1000);
}

void print_inference_result(ei_impulse_result_t result) {

    // Print how long it took to perform inference
    ei_printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
            result.timing.dsp,
            result.timing.classification,
            result.timing.anomaly);

    // Print the prediction results (object detection)
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    ei_printf("Object detection bounding boxes:\r\n");
    for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
    }

    // Print the prediction results (classification)
#else
    ei_printf("Predictions:\r\n");
    int highest_index=0;
    float highest_value=result.classification[0].value;
    //ei_printf("%.5f\r\n", result.classification[1].value);
    //Serial.print(result.classification[0].value);
    for (uint16_t i = 1; i <= EI_CLASSIFIER_LABEL_COUNT; i++) 
      {
        
        if (result.classification[i].value > highest_value) 
          { // If the current value is higher than the highest value
            highest_value = result.classification[i].value;
            highest_index = i;
            //ei_printf("  %s: ", ei_classifier_inferencing_categories[highest_index]);
            //ei_printf("  %s: ", ei_classifier_inferencing_categories[i]);
            //ei_printf("%.5f\r\n", result.classification[i].value);
            //ei_printf("%.5f\r\n", highest_value);
         }
      }
       
       ei_printf("  %s: ", ei_classifier_inferencing_categories[highest_index]);
       ei_printf("%.5f\r\n", highest_value);
       static char predsensor[7];
       dtostrf(highest_value, 6, 2, predsensor);
       client.publish("pred_sensor", predsensor);
       client.publish("pred_cat",ei_classifier_inferencing_categories[highest_index]);
       lcd.setCursor(0,0);
       lcd.print(highest_value);
       lcd.setCursor(0,1);
       lcd.print(ei_classifier_inferencing_categories[highest_index]);
       //lcd.scrollDisplayLeft();
       
       
#endif

    // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

}
