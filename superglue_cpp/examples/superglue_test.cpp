#include "superglue.h"
#include "superpoint.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
    const YAML::Node sp_node = YAML::LoadFile(workspace + "config/superpoint.yaml");
    const YAML::Node sg_node = YAML::LoadFile(workspace + "config/superglue.yaml");
    SuperPoint sp = SuperPoint(sp_node);
    cv::Mat img0 = cv::imread(workspace + "test/data/equirectangular_image_001.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img1 = cv::imread(workspace + "test/data/equirectangular_image_002.jpg", cv::IMREAD_GRAYSCALE);
    // cv::Mat img0 = cv::imread(workspace + "test/data/t/1.png", cv::IMREAD_GRAYSCALE);
    // cv::Mat img1 = cv::imread(workspace + "test/data/t/2.png", cv::IMREAD_GRAYSCALE);
    cv::resize(img0, img0, cv::Size(640, 480), 0, 0);
    cv::resize(img1, img1, cv::Size(640, 480), 0, 0);

    auto [kpts0, desc0] = sp.detect(img0);
    auto [kpts1, desc1] = sp.detect(img1);
    // std::pair<std::vector<cv::KeyPoint>, cv::Mat> temp0 = sp.detect(img0);
    // std::pair<std::vector<cv::KeyPoint>, cv::Mat> temp1 = sp.detect(img1);
    // auto kpts0 = temp0.first;
    // auto kpts1 = temp1.first;
    // auto desc0 = temp0.second;
    // auto desc1 = temp1.second;

    SuperGlue sg = SuperGlue(sg_node);

    for (int i = 0; i < 10; ++i)
    {

        auto [indice0, scores0, indice1, scores1] = sg.match(kpts0, kpts1, desc0, desc1);
        // auto tempp = sg.match(kpts0, kpts1, desc0, desc1);

        for (const auto &point : kpts0)
        {
            cv::circle(img0, point.pt, 2, (255, 255, 255));
        }
        for (const auto &point : kpts1)
        {
            cv::circle(img1, point.pt, 2, (255, 255, 255));
        }

        cv::Mat cat;
        cv::hconcat(img0, img1, cat);
        cv::Mat cat_color = cv::Mat(cat.rows, cat.cols, CV_8UC3);
        cv::cvtColor(cat, cat_color, CV_GRAY2BGR);
        for (int i = 0; i < kpts0.size(); ++i)
        {
            if (indice0[i] == -1)
                continue;

            int color = scores0[i] * 255;
            cv::line(cat_color, kpts0[i].pt, cv::Point(kpts1[indice0[i]].pt.x + img0.cols, kpts1[indice0[i]].pt.y), (255, color, 100));
        }

        cv::imshow("superpoint", cat_color);
        cv::waitKey(20);
    }

    std::cout << "finished!" << std::endl;
    return 0;
}