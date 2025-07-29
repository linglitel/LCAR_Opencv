#include "yolo11.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <set>

static const char* labels[OBJ_CLASS_NUM] = {"Watermelon", "Orange", "Grape", "Apple", "peach", "Banana", "Pineapple"};

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    float w = fmaxf(0.f, fminf(xmax0, xmax1) - fmaxf(xmin0, xmin1) + 1.0f);
    float h = fmaxf(0.f, fminf(ymax0, ymax1) - fmaxf(ymin0, ymin1) + 1.0f);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0f) * (ymax0 - ymin0 + 1.0f) + (xmax1 - xmin1 + 1.0f) * (ymax1 - ymin1 + 1.0f) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float>& outputLocations, const std::vector<int>& classIds,
               std::vector<int>& order,
               int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId) continue;
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId) continue;

            const float* box_n = &outputLocations[n * 4];
            const float* box_m = &outputLocations[m * 4];

            float xmin0 = box_n[0];
            float ymin0 = box_n[1];
            float xmax0 = xmin0 + box_n[2];
            float ymax0 = ymin0 + box_n[3];

            float xmin1 = box_m[0];
            float ymin1 = box_m[1];
            float xmax1 = xmin1 + box_m[2];
            float ymax1 = ymin1 + box_m[3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);
            if (iou > threshold) order[j] = -1;
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices)
{
    if (left >= right) return left;

    int low = left;
    int high = right;
    int key_index = indices[left];
    float key = input[left];

    while (low < high)
    {
        while (low < high && input[high] <= key) high--;
        input[low] = input[high];
        indices[low] = indices[high];
        while (low < high && input[low] >= key) low++;
        input[high] = input[low];
        indices[high] = indices[low];
    }
    input[low] = key;
    indices[low] = key_index;
    quick_sort_indice_inverse(input, left, low - 1, indices);
    quick_sort_indice_inverse(input, low + 1, right, indices);

    return low;
}

inline static float __clip(float val, float min, float max)
{
    return val <= min ? min : (val >= max ? max : val);
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    return (int8_t)__clip(dst_val, -128.f, 127.f);
}

static uint8_t qnt_f32_to_affine_u8(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    return (uint8_t)__clip(dst_val, 0.f, 255.f);
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static void compute_dfl(const float* tensor, int dfl_len, float* box)
{
    for (int b = 0; b < 4; b++)
    {
        const float* src = tensor + b * dfl_len;
        float exp_sum = 0;
        float acc_sum = 0;

        // 第一次循环计算指数和
        for (int i = 0; i < dfl_len; i++)
        {
            float exp_val = expf(src[i]);
            exp_sum += exp_val;
        }

        // 第二次循环计算加权平均
        for (int i = 0; i < dfl_len; i++)
        {
            acc_sum += expf(src[i]) / exp_sum * i;
        }
        box[b] = acc_sum;
    }
}

static int process_u8(const uint8_t* box_tensor, int32_t box_zp, float box_scale,
                      const uint8_t* score_tensor, int32_t score_zp, float score_scale,
                      const uint8_t* score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float>& boxes,
                      std::vector<float>& objProbs,
                      std::vector<int>& classId,
                      float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    uint8_t score_thres_u8 = qnt_f32_to_affine_u8(threshold, score_zp, score_scale);

    if (score_sum_tensor != nullptr)
    {
        uint8_t score_sum_thres_u8 = qnt_f32_to_affine_u8(threshold, score_sum_zp, score_sum_scale);
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                int offset = i * grid_w + j;
                if (score_sum_tensor[offset] < score_sum_thres_u8) continue;

                int max_class_id = -1;
                uint8_t max_score = 0;
                int score_offset = offset;

                for (int c = 0; c < OBJ_CLASS_NUM; c++)
                {
                    uint8_t score = score_tensor[score_offset];
                    if (score > score_thres_u8 && score > max_score)
                    {
                        max_score = score;
                        max_class_id = c;
                    }
                    score_offset += grid_len;
                }

                if (max_score > score_thres_u8)
                {
                    float box[4];
                    float before_dfl[dfl_len * 4];
                    int box_offset = offset;

                    for (int k = 0; k < dfl_len * 4; k++)
                    {
                        before_dfl[k] = deqnt_affine_u8_to_f32(box_tensor[box_offset], box_zp, box_scale);
                        box_offset += grid_len;
                    }
                    compute_dfl(before_dfl, dfl_len, box);

                    float x1 = (-box[0] + j + 0.5f) * stride;
                    float y1 = (-box[1] + i + 0.5f) * stride;
                    float x2 = (box[2] + j + 0.5f) * stride;
                    float y2 = (box[3] + i + 0.5f) * stride;

                    boxes.push_back(x1);
                    boxes.push_back(y1);
                    boxes.push_back(x2 - x1);
                    boxes.push_back(y2 - y1);

                    objProbs.push_back(deqnt_affine_u8_to_f32(max_score, score_zp, score_scale));
                    classId.push_back(max_class_id);
                    validCount++;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                int offset = i * grid_w + j;
                int max_class_id = -1;
                uint8_t max_score = 0;
                int score_offset = offset;

                for (int c = 0; c < OBJ_CLASS_NUM; c++)
                {
                    uint8_t score = score_tensor[score_offset];
                    if (score > score_thres_u8 && score > max_score)
                    {
                        max_score = score;
                        max_class_id = c;
                    }
                    score_offset += grid_len;
                }

                if (max_score > score_thres_u8)
                {
                    float box[4];
                    float before_dfl[dfl_len * 4];
                    int box_offset = offset;

                    for (int k = 0; k < dfl_len * 4; k++)
                    {
                        before_dfl[k] = deqnt_affine_u8_to_f32(box_tensor[box_offset], box_zp, box_scale);
                        box_offset += grid_len;
                    }
                    compute_dfl(before_dfl, dfl_len, box);

                    float x1 = (-box[0] + j + 0.5f) * stride;
                    float y1 = (-box[1] + i + 0.5f) * stride;
                    float x2 = (box[2] + j + 0.5f) * stride;
                    float y2 = (box[3] + i + 0.5f) * stride;

                    boxes.push_back(x1);
                    boxes.push_back(y1);
                    boxes.push_back(x2 - x1);
                    boxes.push_back(y2 - y1);

                    objProbs.push_back(deqnt_affine_u8_to_f32(max_score, score_zp, score_scale));
                    classId.push_back(max_class_id);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

static int process_i8(const int8_t* box_tensor, int32_t box_zp, float box_scale,
                      const int8_t* score_tensor, int32_t score_zp, float score_scale,
                      const int8_t* score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float>& boxes,
                      std::vector<float>& objProbs,
                      std::vector<int>& classId,
                      float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);

    if (score_sum_tensor != nullptr)
    {
        int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                int offset = i * grid_w + j;
                if (score_sum_tensor[offset] < score_sum_thres_i8) continue;

                int max_class_id = -1;
                int8_t max_score = (int8_t)(-score_zp);
                int score_offset = offset;

                for (int c = 0; c < OBJ_CLASS_NUM; c++)
                {
                    int8_t score = score_tensor[score_offset];
                    if (score > score_thres_i8 && score > max_score)
                    {
                        max_score = score;
                        max_class_id = c;
                    }
                    score_offset += grid_len;
                }

                if (max_score > score_thres_i8)
                {
                    float box[4];
                    float before_dfl[dfl_len * 4];
                    int box_offset = offset;

                    for (int k = 0; k < dfl_len * 4; k++)
                    {
                        before_dfl[k] = deqnt_affine_to_f32(box_tensor[box_offset], box_zp, box_scale);
                        box_offset += grid_len;
                    }
                    compute_dfl(before_dfl, dfl_len, box);

                    float x1 = (-box[0] + j + 0.5f) * stride;
                    float y1 = (-box[1] + i + 0.5f) * stride;
                    float x2 = (box[2] + j + 0.5f) * stride;
                    float y2 = (box[3] + i + 0.5f) * stride;

                    boxes.push_back(x1);
                    boxes.push_back(y1);
                    boxes.push_back(x2 - x1);
                    boxes.push_back(y2 - y1);

                    objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                    classId.push_back(max_class_id);
                    validCount++;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                int offset = i * grid_w + j;
                int max_class_id = -1;
                int8_t max_score = (int8_t)(-score_zp);
                int score_offset = offset;

                for (int c = 0; c < OBJ_CLASS_NUM; c++)
                {
                    int8_t score = score_tensor[score_offset];
                    if (score > score_thres_i8 && score > max_score)
                    {
                        max_score = score;
                        max_class_id = c;
                    }
                    score_offset += grid_len;
                }

                if (max_score > score_thres_i8)
                {
                    float box[4];
                    float before_dfl[dfl_len * 4];
                    int box_offset = offset;

                    for (int k = 0; k < dfl_len * 4; k++)
                    {
                        before_dfl[k] = deqnt_affine_to_f32(box_tensor[box_offset], box_zp, box_scale);
                        box_offset += grid_len;
                    }
                    compute_dfl(before_dfl, dfl_len, box);

                    float x1 = (-box[0] + j + 0.5f) * stride;
                    float y1 = (-box[1] + i + 0.5f) * stride;
                    float x2 = (box[2] + j + 0.5f) * stride;
                    float y2 = (box[3] + i + 0.5f) * stride;

                    boxes.push_back(x1);
                    boxes.push_back(y1);
                    boxes.push_back(x2 - x1);
                    boxes.push_back(y2 - y1);

                    objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                    classId.push_back(max_class_id);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

static int process_fp32(const float* box_tensor, const float* score_tensor, const float* score_sum_tensor,
                        int grid_h, int grid_w, int stride, int dfl_len,
                        std::vector<float>& boxes,
                        std::vector<float>& objProbs,
                        std::vector<int>& classId,
                        float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    if (score_sum_tensor != nullptr)
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                int offset = i * grid_w + j;
                if (score_sum_tensor[offset] < threshold) continue;

                int max_class_id = -1;
                float max_score = 0;
                int score_offset = offset;

                for (int c = 0; c < OBJ_CLASS_NUM; c++)
                {
                    float score = score_tensor[score_offset];
                    if (score > threshold && score > max_score)
                    {
                        max_score = score;
                        max_class_id = c;
                    }
                    score_offset += grid_len;
                }

                if (max_score > threshold)
                {
                    float box[4];
                    float before_dfl[dfl_len * 4];
                    int box_offset = offset;

                    for (int k = 0; k < dfl_len * 4; k++)
                    {
                        before_dfl[k] = box_tensor[box_offset];
                        box_offset += grid_len;
                    }
                    compute_dfl(before_dfl, dfl_len, box);

                    float x1 = (-box[0] + j + 0.5f) * stride;
                    float y1 = (-box[1] + i + 0.5f) * stride;
                    float x2 = (box[2] + j + 0.5f) * stride;
                    float y2 = (box[3] + i + 0.5f) * stride;

                    boxes.push_back(x1);
                    boxes.push_back(y1);
                    boxes.push_back(x2 - x1);
                    boxes.push_back(y2 - y1);

                    objProbs.push_back(max_score);
                    classId.push_back(max_class_id);
                    validCount++;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                int offset = i * grid_w + j;
                int max_class_id = -1;
                float max_score = 0;
                int score_offset = offset;

                for (int c = 0; c < OBJ_CLASS_NUM; c++)
                {
                    float score = score_tensor[score_offset];
                    if (score > threshold && score > max_score)
                    {
                        max_score = score;
                        max_class_id = c;
                    }
                    score_offset += grid_len;
                }

                if (max_score > threshold)
                {
                    float box[4];
                    float before_dfl[dfl_len * 4];
                    int box_offset = offset;

                    for (int k = 0; k < dfl_len * 4; k++)
                    {
                        before_dfl[k] = box_tensor[box_offset];
                        box_offset += grid_len;
                    }
                    compute_dfl(before_dfl, dfl_len, box);

                    float x1 = (-box[0] + j + 0.5f) * stride;
                    float y1 = (-box[1] + i + 0.5f) * stride;
                    float x2 = (box[2] + j + 0.5f) * stride;
                    float y2 = (box[3] + i + 0.5f) * stride;

                    boxes.push_back(x1);
                    boxes.push_back(y1);
                    boxes.push_back(x2 - x1);
                    boxes.push_back(y2 - y1);

                    objProbs.push_back(max_score);
                    classId.push_back(max_class_id);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

int post_process(rknn_app_context_t* app_ctx, void* outputs, letterbox_t* letter_box, float conf_threshold,
                 float nms_threshold, object_detect_result_list* od_results)
{
    rknn_output* _outputs = static_cast<rknn_output*>(outputs);
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;

    memset(od_results, 0, sizeof(object_detect_result_list));

    int dfl_len = app_ctx->output_attrs[0].dims[1] / 4;
    int output_per_branch = app_ctx->io_num.n_output / 3;

    for (int i = 0; i < 3; i++)
    {
        const void* score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0f;

        if (output_per_branch == 3)
        {
            score_sum = _outputs[i * output_per_branch + 2].buf;
            score_sum_zp = app_ctx->output_attrs[i * output_per_branch + 2].zp;
            score_sum_scale = app_ctx->output_attrs[i * output_per_branch + 2].scale;
        }

        int box_idx = i * output_per_branch;
        int score_idx = i * output_per_branch + 1;
        grid_h = app_ctx->output_attrs[box_idx].dims[2];
        grid_w = app_ctx->output_attrs[box_idx].dims[3];
        stride = model_in_h / grid_h;

        if (app_ctx->is_quant)
        {
            validCount += process_i8(static_cast<const int8_t*>(_outputs[box_idx].buf), app_ctx->output_attrs[box_idx].zp,
                                     app_ctx->output_attrs[box_idx].scale,
                                     static_cast<const int8_t*>(_outputs[score_idx].buf), app_ctx->output_attrs[score_idx].zp,
                                     app_ctx->output_attrs[score_idx].scale,
                                     static_cast<const int8_t*>(score_sum), score_sum_zp, score_sum_scale,
                                     grid_h, grid_w, stride, dfl_len,
                                     filterBoxes, objProbs, classId, conf_threshold);
        }
        else
        {
            validCount += process_fp32(static_cast<const float*>(_outputs[box_idx].buf), static_cast<const float*>(_outputs[score_idx].buf),
                                       static_cast<const float*>(score_sum),
                                       grid_h, grid_w, stride, dfl_len,
                                       filterBoxes, objProbs, classId, conf_threshold);
        }
    }

    if (validCount <= 0) return 0;

    std::vector<int> indexArray(validCount);
    for (int i = 0; i < validCount; ++i) indexArray[i] = i;
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(classId.begin(), classId.end());
    for (auto c : class_set) nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);

    int last_count = 0;
    od_results->count = 0;

    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) continue;
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        od_results->results[last_count].box.left = static_cast<int>(clamp(x1, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.top = static_cast<int>(clamp(y1, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.right = static_cast<int>(clamp(x2, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.bottom = static_cast<int>(clamp(y2, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}

int init_post_process() { return 0; }

const char* coco_cls_to_name(int cls_id)
{
    if (cls_id >= OBJ_CLASS_NUM) return "null";
    return labels[cls_id] ? labels[cls_id] : "null";
}
