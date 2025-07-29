//
// Created by ling on 2025/7/29.
//

#ifndef LCAR_OPENCV_YOLO11_H
#define LCAR_OPENCV_YOLO11_H

#include <vector>
#include <cmath>
#include <stdint.h>
#include <cstring>
#include "rknn_api.h"
#include <opencv2/opencv.hpp>
#include <set>

#define OBJ_CLASS_NUM 7
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_NAME_MAX_SIZE 64
#define NMS_THRESH 0.45f
#define BOX_THRESH 0.25f

typedef struct
{
    int left;
    int top;
    int right;
    int bottom;
} image_rect_t;

typedef struct
{
    image_rect_t box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct
{
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

typedef struct {
    int x_pad;
    int y_pad;
    float scale;
} letterbox_t;

typedef enum
{
    IMAGE_FORMAT_GRAY8,
    IMAGE_FORMAT_RGB888,
    IMAGE_FORMAT_RGBA8888,
    IMAGE_FORMAT_YUV420SP_NV21,
    IMAGE_FORMAT_YUV420SP_NV12,
} image_format_t;

typedef struct
{
    int width;
    int height;
    int width_stride;
    int height_stride;
    image_format_t format;
    unsigned char* virt_addr;
    int size;
    int fd;
} image_buffer_t;

typedef struct
{
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_app_context_t;

int init_yolo11_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_yolo11_model(rknn_app_context_t* app_ctx);
int inference_yolo11_model(rknn_app_context_t* app_ctx, cv::Mat& img, object_detect_result_list* od_results);

// 后处理相关
int init_post_process();
const char* coco_cls_to_name(int cls_id);
int post_process(rknn_app_context_t* app_ctx, void* outputs, letterbox_t* letter_box, float conf_threshold,
                 float nms_threshold, object_detect_result_list* od_results);


#endif //LCAR_OPENCV_YOLO11_H
