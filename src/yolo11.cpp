#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include "yolo11.h"
#include <vector>
#include <memory>
#include <fstream>
#include <cstring>

// 使用智能指针管理内存
using TensorAttrPtr = std::unique_ptr<rknn_tensor_attr[]>;

void convert_image_with_letterbox(const cv::Mat& src, cv::Mat& dst, letterbox_t& lb, int bg_color)
{
    int in_h = src.rows;
    int in_w = src.cols;
    int out_h = dst.rows;
    int out_w = dst.cols;

    float ratio = std::min((float)out_w / (float)in_w, (float)out_h / (float)in_h);

    int new_w = static_cast<int>(in_w * ratio);
    int new_h = static_cast<int>(in_h * ratio);

    int pad_w = (out_w - new_w) / 2;
    int pad_h = (out_h - new_h) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    dst = cv::Scalar(bg_color, bg_color, bg_color);

    cv::Rect roi(pad_w, pad_h, new_w, new_h);
    resized.copyTo(dst(roi));

    lb.scale = ratio;
    lb.y_pad = pad_h;
    lb.x_pad = pad_w;
}

int init_yolo11_model(const char* model_path, rknn_app_context_t* app_ctx)
{
    if (!app_ctx) return -1;

    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return -1;

    size_t model_len = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> model(model_len);
    file.read(model.data(), model_len);
    file.close();

    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model.data(), model_len, 0, nullptr);
    if (ret < 0) return -1;

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) return -1;

    auto input_attrs = std::make_unique<rknn_tensor_attr[]>(io_num.n_input);
    for (int i = 0; i < io_num.n_input; ++i)
    {
        input_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
    }

    auto output_attrs = std::make_unique<rknn_tensor_attr[]>(io_num.n_output);
    for (int i = 0; i < io_num.n_output; ++i)
    {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
    }

    app_ctx->rknn_ctx = ctx;
    app_ctx->is_quant = (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
        output_attrs[0].type == RKNN_TENSOR_INT8);

    app_ctx->io_num = io_num;
    app_ctx->input_attrs = input_attrs.release();
    app_ctx->output_attrs = output_attrs.release();

    const auto& input_attr = app_ctx->input_attrs[0];
    if (input_attr.fmt == RKNN_TENSOR_NCHW)
    {
        app_ctx->model_channel = input_attr.dims[1];
        app_ctx->model_height = input_attr.dims[2];
        app_ctx->model_width = input_attr.dims[3];
    }
    else
    {
        app_ctx->model_height = input_attr.dims[1];
        app_ctx->model_width = input_attr.dims[2];
        app_ctx->model_channel = input_attr.dims[3];
    }

    return 0;
}

int release_yolo11_model(rknn_app_context_t* app_ctx)
{
    if (!app_ctx) return -1;

    if (app_ctx->input_attrs)
    {
        delete[] app_ctx->input_attrs;
        app_ctx->input_attrs = nullptr;
    }

    if (app_ctx->output_attrs)
    {
        delete[] app_ctx->output_attrs;
        app_ctx->output_attrs = nullptr;
    }

    if (app_ctx->rknn_ctx)
    {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }

    return 0;
}

int inference_yolo11_model(rknn_app_context_t* app_ctx, cv::Mat& img, object_detect_result_list* od_results)
{
    if (!app_ctx || img.empty() || !od_results) return -1;

    std::memset(od_results, 0, sizeof(*od_results));

    letterbox_t letter_box{};
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    const int bg_color = 114;

    cv::Mat resized_img(app_ctx->model_height, app_ctx->model_width, CV_8UC3);
    convert_image_with_letterbox(img, resized_img, letter_box, bg_color);

    rknn_input inputs[1] = {};
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = resized_img.total() * resized_img.elemSize();
    inputs[0].buf = resized_img.data;

    int ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, inputs);
    if (ret < 0) return -1;

    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) return -1;

    std::vector<rknn_output> outputs(app_ctx->io_num.n_output);
    for (int i = 0; i < app_ctx->io_num.n_output; ++i)
    {
        outputs[i].index = i;
        outputs[i].want_float = !app_ctx->is_quant;
    }

    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs.data(), nullptr);
    if (ret < 0) return -1;

    post_process(app_ctx, outputs.data(), &letter_box, box_conf_threshold, nms_threshold, od_results);

    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs.data());

    return 0;
}
