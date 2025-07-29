#include <opencv2/opencv.hpp>
#include "yolo11.h"


int main()
{
    // 打开摄像头
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        printf("Cannot open camera\n");
        return -1;
    }

    // 初始化模型
    rknn_app_context_t app_ctx = {0};
    if (init_yolo11_model("/home/ling/CLionProjects/LCAR_Opencv/best.rknn", &app_ctx) < 0)
    {
        printf("Init model failed\n");
        return -1;
    }

    cv::Mat frame;
    object_detect_result_list results;


    while (true)
    {
        // 读取摄像头帧
        cap >> frame;
        if (frame.empty()) break;

        // 推理
        memset(&results, 0, sizeof(results));
        if (inference_yolo11_model(&app_ctx, frame, &results) == 0)
        {
            if (results.count > 0)
            {
                std::cout << "Detected " << results.count << " objects:" << std::endl;
                for (int i = 0; i < results.count; i++)
                {
                    object_detect_result* det = &results.results[i];
                    const char* class_name = coco_cls_to_name(det->cls_id);
                    cv::rectangle(frame,
                              cv::Point(det->box.left, det->box.top),
                              cv::Point(det->box.right, det->box.bottom),
                              cv::Scalar(0, 255, 0), 2);

                    // 画标签
                    std::string label = (class_name ? class_name : "Unknown") + std::to_string(det->prop).substr(0, 4);
                    int baseLine;
                    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                    cv::rectangle(frame,
                                  cv::Point(det->box.left, det->box.top - labelSize.height - 10),
                                  cv::Point(det->box.left + labelSize.width, det->box.top),
                                  cv::Scalar(0, 255, 0), -1);

                    cv::putText(frame, label,
                                cv::Point(det->box.left, det->box.top - 5),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

                }
                std::cout << "---" << std::endl;
            }
            else
            {
            }
        }
        cv::imshow("Detection Result", frame);
        // 按ESC退出
        if (cv::waitKey(1) == 27) break;
    }

    // 释放资源
    release_yolo11_model(&app_ctx);
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
