#include "config.h"
#include "superpoint.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

int main()
{
    const YAML::Node node = YAML::LoadFile(workspace + "config/superpoint.yaml");
    SuperPoint sp = SuperPoint(node);

    cv::namedWindow("superpoint", cv::WINDOW_AUTOSIZE);
    for (int i = 0; i < 10; ++i)
    {
        cv::Mat img = cv::imread(workspace + "test/data/equirectangular_image_001.jpg", cv::IMREAD_GRAYSCALE);
        auto points_and_desc = sp.detect(img);
        for (const auto &point : points_and_desc.first)
        {
            cv::circle(img, point.pt, 2, (255, 255, 255));
        }

        cv::imshow("superpoint", img);
        cv::waitKey(20);
    }
    cv::Mat img = cv::imread(workspace + "test/data/equirectangular_image_001.jpg", cv::IMREAD_GRAYSCALE);
    cv::resize(img, img, cv::Size(640, 480), 0, 0);
    auto points_and_desc = sp.detect(img);
    for (const auto &point : points_and_desc.first)
    {
        cv::circle(img, point.pt, 2, (255, 255, 255));
    }

    cv::imshow("superpoint", img);
    cv::waitKey(20);

    std::cout << "finshed" << std::endl;
    return 0;
}