#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <tensorflow/c/c_api.h>

#define NUM_WARMUP_RUNS 100
#define NUM_ACTUAL_RUNS 100

void DeallocateBuffer(void* data, size_t length, void* arg) {
    free(data);
}

void PrintTensorInfo(const char* name, TF_Tensor* tensor) {
    if (tensor == NULL) {
        printf("%s is NULL\n", name);
        return;
    }
    int num_dims = TF_NumDims(tensor);
    printf("%s: %d dimensions, type: %d\n", name, num_dims, TF_TensorType(tensor));
    for (int i = 0; i < num_dims; i++) {
        printf("Dim %d: %ld\n", i, TF_Dim(tensor,  i));
    }
}

double Run(TF_Session* session, TF_Status* status, TF_Output* input_op, TF_Output* output_op, TF_Tensor* input_tensor) {
    if (session == NULL || status == NULL || input_op == NULL || output_op == NULL || input_tensor == NULL) {
        fprintf(stderr, "Invalid arguments to Run\n");
        return -1;
    }

    TF_Tensor* output_tensor = NULL;
    clock_t start = clock();

    TF_SessionRun(session, NULL, input_op, &input_tensor, 1, output_op, &output_tensor, 1, NULL, 0, NULL, status);
    
    clock_t end = clock();
    
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error running session: %s\n", TF_Message(status));
        return -1;
    }

    if (output_tensor != NULL) {
        TF_DeleteTensor(output_tensor);
    }
    
    return ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0; 
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <path_to_saved_model> <seq_len>\n", argv[0]);
        return 1;
    }

    const char* saved_model_dir = argv[1];
    int seq_len = atoi(argv[2]);

    // Load the saved model
    TF_Graph* graph = TF_NewGraph();
    TF_Status* status = TF_NewStatus();
    TF_SessionOptions* session_opts = TF_NewSessionOptions();
    uint8_t config[] = {0x10, 0x1, 0x28, 0x1}; // {0x10, 0x01}; // Protobuf-encoded ConfigProto with intra_op_parallelism_threads = 1
    TF_SetConfig(session_opts, (void*)config, sizeof(config), status);
    TF_Buffer* run_opts = NULL;
    const char* tags = "serve";
    int ntags = 1;
    TF_Session* session = TF_LoadSessionFromSavedModel(session_opts, run_opts, saved_model_dir, &tags, ntags, graph, NULL, status);

    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error loading saved model: %s\n", TF_Message(status));
        return 1;
    }

    // Get input and output operations
    TF_Output input_op = {TF_GraphOperationByName(graph, "serving_default_args_0"), 0};
    TF_Output output_op = {TF_GraphOperationByName(graph, "StatefulPartitionedCall_1"), 0};

    if (input_op.oper == NULL || output_op.oper == NULL) {
        fprintf(stderr, "Error getting input or output operations\n");
        return 1;
    }

    // Prepare input data
    const int64_t input_dims[] = {1, seq_len, 5}; // [batch_size, seq_len, num_features]
    const int num_elements = seq_len * 5; // 1 * seq_len * 5
    float* input_data = (float*)malloc(num_elements * sizeof(float));
    if (input_data == NULL) {
        fprintf(stderr, "Failed to allocate memory for input data\n");
        return 1;
    }
    for (int i = 0; i < num_elements; ++i) {
        input_data[i] = (float)rand() / RAND_MAX;
    }

    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, input_dims, 3, input_data, num_elements * sizeof(float), DeallocateBuffer, NULL);
    if (input_tensor == NULL) {
        fprintf(stderr, "Failed to create input tensor\n");
        free(input_data);
        return 1;
    }

    // Warmup runs
    printf("Performing warmup runs...\n");
    for (int i = 0; i < NUM_WARMUP_RUNS; i++) {
        double run_time = Run(session, status, &input_op, &output_op, input_tensor);
        if (run_time < 0) {
            fprintf(stderr, "Error during warmup run %d\n", i);
            TF_DeleteTensor(input_tensor);
            return 1;
        }
    }

    // Actual profiling runs
    printf("Performing actual profiling runs...\n");
    double total_time = 0.0;
    double min_time = DBL_MAX;
    double max_time = 0.0;
    for (int i = 0; i < NUM_ACTUAL_RUNS; i++) {
        double run_time = Run(session, status, &input_op, &output_op, input_tensor);
        if (run_time < 0) {
            fprintf(stderr, "Error during inference run %d\n", i);
            TF_DeleteTensor(input_tensor);
            return 1;
        }
        total_time += run_time;
        if (run_time < min_time) min_time = run_time;
        if (run_time > max_time) max_time = run_time;
    }


    // Calculate and print results
    double average_time = total_time / NUM_ACTUAL_RUNS;
    printf("Inference time over %d runs:\n", NUM_ACTUAL_RUNS);
    printf("  Average: %.3f ms\n", average_time);
    printf("  Min: %.3f ms\n", min_time);
    printf("  Max: %.3f ms\n", max_time);

    // Clean up
    TF_DeleteTensor(input_tensor);
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(session_opts);
    TF_DeleteStatus(status);
    TF_DeleteGraph(graph);

    return 0;
}