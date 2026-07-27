#include <math_constants.h>

extern "C" __global__ void update_gmm(
    const float* frame,           // (3, H, W)
    const float* diff_square_sum, // (K, H, W)
    float* means,                 // (K, 3, H, W)
    float* vars,                  // (K, H, W)
    float* weights,               // (K, H, W)
    float match_threshold,
    float update_alpha,
    float INIT_VAR,
    float REINIT_WEIGHT,
    int num_pixels, int C, int K
) {
    // Thread coordinating values
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Pre-compute constants
    float sqr_threshold = match_threshold * match_threshold;
    float one_minus_alpha = 1.0 - update_alpha;

    for (int i = idx; i < num_pixels; i += stride) {
        
        // Find best matching component
        int matched_comp_idx = -1;
        float min_err = CUDART_INF_F;
        
        for (int k = 0; k < K; k++) {
            float diff = diff_square_sum[num_pixels*k + i];
            float bound = vars[num_pixels*k + i] * sqr_threshold;

            if (diff < bound && diff < min_err) {
                matched_comp_idx = k;
                min_err = diff;
            }
        }

        // Update statistics
        for (int k = 0; k < K; k++) {
            // Decay weight of all components
            long long comp_idx = (long long) num_pixels*k + i;
            weights[comp_idx] = weights[comp_idx] * one_minus_alpha;
        }

        if (matched_comp_idx != -1) {
            // Update weight and variance
            long long weight_var_idx = (long long) num_pixels*matched_comp_idx + i;
            weights[weight_var_idx] += update_alpha;
            vars[weight_var_idx] = one_minus_alpha*vars[weight_var_idx] + update_alpha*min_err;

            // Update mean
            for (int c = 0; c < C; c++) {
                long long mean_idx = (long long) num_pixels*(matched_comp_idx*C + c) + i;
                long long frame_idx = (long long) num_pixels*c + i;

                means[mean_idx] = one_minus_alpha*means[mean_idx] + update_alpha*frame[frame_idx];
            }
        } else {
            int weakest_comp_idx = -1;
            float weakest_weight = CUDART_INF_F;

            // Find the component of smallest weight
            for (int k = 0; k < K; k++) {
                long long comp_idx = (long long) num_pixels*k + i;
                if (weights[comp_idx] < weakest_weight) {
                    weakest_comp_idx = k;
                    weakest_weight = weights[comp_idx];
                }
            }

            // Replace the weakest component's statistics

            // Re-initialize weight and variance
            long long weight_var_idx = (long long) num_pixels*weakest_comp_idx + i;
            weights[weight_var_idx] = REINIT_WEIGHT;
            vars[weight_var_idx] = INIT_VAR;

            for (int c = 0; c < C; c++) {
                //Replace mean
                long long mean_idx = (long long) num_pixels*(weakest_comp_idx*C + c) + i;   
                long long frame_idx = (long long) num_pixels*c + i;

                means[mean_idx] = frame[frame_idx];
            }
        }

        // Normalize weights
        float w_sum = 0.0;
        for (int k = 0; k < K; k++) {
            w_sum += weights[num_pixels*k + i];
        }
        for (int k = 0; k < K; k++) {
            int w_idx = num_pixels*k + i;
            weights[w_idx] = weights[w_idx] / w_sum;
        }
    }
}