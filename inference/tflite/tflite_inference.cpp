#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <limits>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

#define NUM_WARMUP_RUNS 5
#define NUM_ACTUAL_RUNS 100

void fill_random(std::vector<float>& data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (auto& val : data) {
        val = dis(gen);
    }
}

double run_inference(std::unique_ptr<tflite::Interpreter>& interpreter) {
    if (!interpreter) {
        std::cerr << "Invalid interpreter" << std::endl;
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Error invoking interpreter" << std::endl;
        return -1;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_tflite_model>" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];

    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    // Create XNNPACK delegate
    auto xnnpack_options = TfLiteXNNPackDelegateOptionsDefault();
    xnnpack_options.num_threads = 1;
    auto xnnpack_delegate = TfLiteXNNPackDelegateCreate(&xnnpack_options);

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    
    builder.AddDelegate(xnnpack_delegate);
    builder(&interpreter);

    if (!interpreter || interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to create interpreter or allocate tensors" << std::endl;
        TfLiteXNNPackDelegateDelete(xnnpack_delegate);
        return 1;
    }

    // Get input tensor
    int input = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input);
    
    // Fill input tensor with random data
    std::vector<float> input_data(input_tensor->bytes / sizeof(float));
    fill_random(input_data);
    std::memcpy(input_tensor->data.f, input_data.data(), input_data.size() * sizeof(float));

    // Warmup runs
    std::cout << "Performing warmup runs..." << std::endl;
    for (int i = 0; i < NUM_WARMUP_RUNS; ++i) {
        double run_time = run_inference(interpreter);
        if (run_time < 0) {
            std::cerr << "Error during warmup run " << i << std::endl;
            TfLiteXNNPackDelegateDelete(xnnpack_delegate);
            return 1;
        }
    }
    // Actual profiling runs
    std::cout << "Performing actual profiling runs..." << std::endl;
    double total_time = 0.0;
    double min_time = std::numeric_limits<double>::max();
    double max_time = 0.0;
    for (int i = 0; i < NUM_ACTUAL_RUNS; ++i) {
        double run_time = run_inference(interpreter);
        if (run_time < 0) {
            std::cerr << "Error during inference run " << i << std::endl;
            TfLiteXNNPackDelegateDelete(xnnpack_delegate);
            return 1;
        }
        total_time += run_time;
        min_time = std::min(min_time, run_time);
        max_time = std::max(max_time, run_time);
    }

    // Calculate and print results
    double average_time = total_time / NUM_ACTUAL_RUNS;
    std::cout << "Inference time over " << NUM_ACTUAL_RUNS << " runs:" << std::endl;
    std::cout << "  Average: " << average_time << " ms" << std::endl;
    std::cout << "  Min: " << min_time << " ms" << std::endl;
    std::cout << "  Max: " << max_time << " ms" << std::endl;

    // Clean up
    TfLiteXNNPackDelegateDelete(xnnpack_delegate);

    return 0;
}
