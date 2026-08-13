#define N_CHANNELs 3

__device__ bool detect_shadow(
    const int i,
    const int nmodes,
    const float* frame,
    const float* weights,
    const float* means,
    const float* vars_,
    const float Tb,
    const float TB,
    const float tau,
    const int num_pixels, const int C) {

    float t_weight = 0.0;
    for (int mode = 0; mode < nmodes; mode++) {
        float num = 0.0;
        float den = 0.0;

        for (int c = 0; c < C; c++) {
            long long mean_idx = (long long) ((mode*C + c)*num_pixels + i);
            long long frame_idx = (long long) (c*num_pixels + i);
            
            num += frame[frame_idx] * means[mean_idx];
            den += means[mean_idx] * means[mean_idx];
        }

        if (den == 0.0)
            return false;
        
        long long w_v_idx = (long long) (mode*num_pixels + i);
        if (num <= den && num >= tau*den) {
            float a = num / den;
            float dist2a = 0.0;
            for (int c = 0; c < C; c++) {
                long long mean_idx = (long long) ((mode*C + c)*num_pixels + i);
                long long frame_idx = (long long) (c*num_pixels + i);

                float diff = a*means[mean_idx] - frame[frame_idx];
                dist2a += diff*diff;
            }
            
            if (dist2a < Tb*vars_[w_v_idx]*a*a)
                return true;
        }
        t_weight += weights[w_v_idx];
        if (t_weight > TB)
            return false;
    }
    return false;
}

extern "C" __global__ void step_gmm(
    const float* frame,
    float* weights,
    float* means,
    float* vars_,
    unsigned char* modes,
    unsigned char* mask,
    float* bg_prob,
    const float FLT_EPS,
    const int H, const int W, const int C, const int K,
    const float alpha,
    const float prune,
    const float Tb,
    const float Tg,
    const float TB,
    const float var_init,
    const float var_min,
    const float var_max,
    const float tau,
    const unsigned char shadow_val,
    const bool detect_shadows) {
        // Thread coordinating values
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        int stride_x = blockDim.x * gridDim.x;
        int stride_y = blockDim.y * gridDim.y;

        // Pre-computed constant
        int num_pixels = H * W;
        float alpha1 = 1.0 - alpha;

        // Local buffer
        float dData[N_CHANNELs];
        float pData[N_CHANNELs];

        for (int yy = y; yy < H; yy += stride_y)
            for (int xx = x; xx < W; xx += stride_x) {
                int i = yy * W + xx;
                bool background = false;
                bool fit_pdf = false;
                unsigned char nmodes = modes[i];
                
                for (int c = 0; c < C; c++) {
                    long long frame_idx = (long long) (c*num_pixels + i);
                    pData[c] = frame[frame_idx];
                }


                float total_weight = 0.0;
                float bg_weight_sum = 0.0f;
                for (int mode = 0; mode < nmodes; mode++) {
                    long long w_v_idx = (long long) (mode*num_pixels + i);
                    float weight = alpha1 * weights[w_v_idx] + prune;
                    int swap_count = 0;

                    if (!fit_pdf) {
                        float var = vars_[w_v_idx];
                        float dist2 = 0.0;

                        for (int c = 0; c < C; c++) {
                            long long mean_idx = (long long) ((mode*C + c)*num_pixels + i);
                            float diff = means[mean_idx] - pData[c];
                            dData[c] = diff;
                            dist2 += diff * diff;
                        }

                        if ((total_weight < TB) && (dist2 < Tb * var)) {
                            background = true;
                            bg_weight_sum += weights[w_v_idx];
                        }

                        if (dist2 < Tg * var) {
                            fit_pdf = true;
                            weight += alpha;
                            float k = alpha / weight;

                            for (int c = 0; c < C; c++) {
                                long long mean_idx = (long long) ((mode*C + c)*num_pixels + i);
                                means[mean_idx] -= k * dData[c];
                            }
                            
                            float varnew = var + k*(dist2 - var);
                            varnew = max(varnew, var_min);
                            varnew = min(varnew, var_max);
                            vars_[w_v_idx] = varnew;
                            
                            float temp = 0.0;
                            for (int u = mode; u > 0; u--) {
                                long long w_v_prev_idx = (long long) ((u-1)*num_pixels + i);
                                if (weight < weights[w_v_prev_idx])
                                    break;
                                swap_count += 1;

                                long long w_v_cur_idx = (long long) (u*num_pixels + i);

                                temp = weights[w_v_cur_idx];
                                weights[w_v_cur_idx] = weights[w_v_prev_idx];
                                weights[w_v_prev_idx] = temp;
                                
                                temp = vars_[w_v_cur_idx];
                                vars_[w_v_cur_idx] = vars_[w_v_prev_idx];
                                vars_[w_v_prev_idx] = temp;

                                for (int c = 0; c < C; c++) {
                                    long long mean_cur_idx = (long long) ((u*C + c)*num_pixels + i);
                                    long long mean_prev_idx = (long long) (((u-1)*C + c)*num_pixels + i);

                                    temp = means[mean_cur_idx];
                                    means[mean_cur_idx] = means[mean_prev_idx];
                                    means[mean_prev_idx] = temp;
                                }
                            }
                        }

                    }
                
                    if (weight < -prune) {
                        // weight = 0.0;
                        // nmodes -= 1;
                    }

                    weights[(long long) ((mode-swap_count)*num_pixels + i)] = weight;
                    total_weight += weight;
                }

                float inv_weight = 0.0;
                if (fabsf(total_weight) > FLT_EPS)
                    inv_weight = 1.0 / total_weight;

                for (int mode = 0; mode < nmodes; mode++) {
                    long long w_idx = (long long) (mode*num_pixels + i);
                    weights[w_idx] *= inv_weight;
                }

                if (!fit_pdf && alpha > 0.0) {
                    int mode = 0;
                    if (nmodes == K) {
                        mode = K - 1;
                    } else {
                        mode = nmodes;
                        nmodes += 1;
                    }

                    if (nmodes == 1) {
                        weights[(long long) (mode*num_pixels + i)] = 1.0;
                    } else {
                        weights[(long long) (mode*num_pixels + i)] = alpha;
                        for (int m = 0; m < nmodes - 1; m++) {
                            weights[(long long) (m*num_pixels + i)] *= alpha1;
                        }
                    }

                    for (int c = 0; c < C; c++) {
                        long long mean_idx = (long long) ((mode*C + c)*num_pixels + i);   
                        means[mean_idx] = pData[c];
                    }

                    vars_[(long long) (mode*num_pixels + i)] = var_init;

                    float temp = 0.0;
                    for (int u = nmodes - 1; u > 0; u--) {
                        long long w_v_prev_idx = (long long) ((u-1)*num_pixels + i);
                        if (alpha < weights[w_v_prev_idx])
                            break;

                        long long w_v_cur_idx = (long long) (u*num_pixels + i);

                        temp = weights[w_v_cur_idx];
                        weights[w_v_cur_idx] = weights[w_v_prev_idx];
                        weights[w_v_prev_idx] = temp;
                        
                        temp = vars_[w_v_cur_idx];
                        vars_[w_v_cur_idx] = vars_[w_v_prev_idx];
                        vars_[w_v_prev_idx] = temp;

                        for (int c = 0; c < C; c++) {
                            long long mean_cur_idx = (long long) ((u*C + c)*num_pixels + i);
                            long long mean_prev_idx = (long long) (((u-1)*C + c)*num_pixels + i);

                            temp = means[mean_cur_idx];
                            means[mean_cur_idx] = means[mean_prev_idx];
                            means[mean_prev_idx] = temp;
                        }
                    }
                }
                
                modes[i] = nmodes;
                bg_prob[i] = bg_weight_sum;

                if (background) {
                    mask[i] = 0;
                } else if (detect_shadows && detect_shadow(i, nmodes, frame, weights, means, vars_, Tb, TB, tau, num_pixels, C)) {
                    mask[i] = shadow_val;
                } else {
                    mask[i] = 255;
                }
            }
        
    }