#ifndef OUTPUT_HANDLER_H
#define OUTPUT_HANDLER_H

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"

void handleOutput(
    tflite::ErrorReporter* error_reporter,
    float x_value,
    float y_value
);

#endif
