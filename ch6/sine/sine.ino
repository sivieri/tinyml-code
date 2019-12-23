#include <TensorFlowLite.h>

#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "sine_model_quantized.hh"
#include "constants.hh"
#include "output_handler.hh"

namespace {
    tflite::ErrorReporter* error_reporter = nullptr;
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    int inference_count = 0;

    constexpr int kTensorArenaSize = 2 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];
}

void setup() {
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    model = tflite::GetModel(sine_model_quantized_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report(
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            model->version(),
            TFLITE_SCHEMA_VERSION
        );
        return;
    }

    static tflite::ops::micro::AllOpsResolver resolver;

    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize,
        error_reporter
    );
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        error_reporter->Report("AllocateTensors() failed");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    inference_count = 0;
}

void loop() {
    float position = static_cast<float>(inference_count) / static_cast<float>(kInferencesPerCycle);
    float x_val = position * kXrange;

    input->data.f[0] = x_val;

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        error_reporter->Report(
            "Invoke failed on x_val: %f\n",
            static_cast<double>(x_val)
        );
        return;
    }

    float y_val = output->data.f[0];

    handleOutput(error_reporter, x_val, y_val);

    inference_count += 1;
    if (inference_count >= kInferencesPerCycle) inference_count = 0;
}
