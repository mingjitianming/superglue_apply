#include "superpoint.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
    const YAML::Node node = YAML::Node();
    SuperPoint sp = SuperPoint(node);
    cv::Mat img = cv::imread("/home/zmy/project_ws/superglue_apply/superglue_cpp/test/data/equirectangular_image_001.jpg", cv::IMREAD_GRAYSCALE);
    cv::resize(img, img, cv::Size(640, 480), 0, 0);
    sp.detect(img);
    std::cout << "finshed" << std::endl;
    return 0;
}