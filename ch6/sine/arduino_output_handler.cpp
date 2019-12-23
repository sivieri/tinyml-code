#include "Arduino.h"

#include "output_handler.hh"

int led = LED_BUILTIN;
bool initialized = false;

void handleOutput(
    tflite::ErrorReporter* error_reporter,
    float x_value,
    float y_value
) {
    if (!initialized) {
        pinMode(led, OUTPUT);
        initialized = true;
    }
    int brightness = (int)(127.5f * (y_value + 1));
    analogWrite(led, brightness);
    error_reporter->Report("%d\n", brightness);
}
