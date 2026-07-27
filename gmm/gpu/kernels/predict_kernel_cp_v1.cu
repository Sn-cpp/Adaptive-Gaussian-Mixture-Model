#define MAX_K 20

extern "C" __global__ void predict_gmm(
    const float* frame,           // (3, H, W)
    float* diff_square_sum,       // (K, H, W)
    unsigned char* mask,                // (H, W)
    float* means,                 // (K, 3, H, W)
    float* vars,                  // (K, H, W)
    float* weights,               // (K, H, W)
    float match_threshold,
    float background_threshold,
    int num_pixels, int C, int K
) {
    // Thread coordinating values
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Pre-compute constants
    float sqr_match = match_threshold * match_threshold;

    // Temporary variables
    float ranks[MAX_K];
    int order[MAX_K];
    bool matches[MAX_K];

    for (int i = idx; i < num_pixels; i += stride) {
        for (int k = 0; k < K; k++) {
            float diff_sum = 0.0;

            // Manually compute Euclidean distance between pixel and component k
            for (int c = 0; c < C; c++) {
                float diff = frame[num_pixels*c + i] - means[num_pixels*(k*C + c) + i];
                diff_sum += (diff * diff);
            }
            // Compute re-usable index
            int khw_i = num_pixels*k + i;

            // Record diff square value
            diff_square_sum[khw_i] = diff_sum;
            
            // Check if this component is matched
            matches[k] = diff_sum < (sqr_match * vars[khw_i]);
            
            // Compute ranking value
            ranks[k] = weights[khw_i] / (sqrt(vars[khw_i]) + 1e-6);
        }

        for (int k = 0; k < K; k++) {
            order[k] = k;
        }
                
        // Utilize a sort algorithm
        for (int u = 0; u < K - 1; u++) {
            for (int v = 0; v < K - 1 - u; v++) {
                if (ranks[order[v]] < ranks[order[v + 1]]) {
                    int tmp = order[v];
                    order[v] = order[v + 1];
                    order[v + 1] = tmp;
                }
            }
        }

        // Determine background/foreground
        bool is_background = false;
        double cumulative_weight = 0.0;

        for (int k = 0; k < K; k++) {
            cumulative_weight += weights[num_pixels*order[k] + i];

            // Use cumulative relative weight and threshold to determine the background
            bool included = (k == 0 || cumulative_weight <= background_threshold);
            if (!included)
                break; // Exceeding threshold
            
            if (matches[order[k]]) { // Must be a matched component
                is_background = true;
                break; // Force-stop to save time
            } 
        }
        mask[i] = is_background ? 0 : 255;          
    }
}