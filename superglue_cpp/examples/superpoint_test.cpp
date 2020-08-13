#include "superpoint.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "config.h"

int main()
{
    std::string workspace = WORKSPACE_DIR;
    const YAML::Node node = YAML::LoadFile(workspace + "config/superpoint.yaml");
    SuperPoint sp = SuperPoint(node);
    cv::Mat img = cv::imread(workspace + "test/data/equirectangular_image_001.jpg", cv::IMREAD_GRAYSCALE);
    cv::resize(img, img, cv::Size(640, 480), 0, 0);
    sp.detect(img);
    std::cout << "finshed" << std::endl;
    return 0;
}