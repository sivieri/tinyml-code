#include "output_handler.hh"

void handleOutput(
    tflite::ErrorReporter* error_reporter,
    float x_value,
    float y_value
) {
    error_reporter->Report("x_value: %f, y_value: %f\n", x_value, y_value);
}
