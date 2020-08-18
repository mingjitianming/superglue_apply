#ifndef SUPERPOINT_H
#define SUPERPOINT_H

#include "config.h"
#include <memory>
#include <opencv2/core/core.hpp>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

class SuperPoint
{

public:
    SuperPoint(const YAML::Node &config_node);
    ~SuperPoint() = default;
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> detect(const cv::Mat &image);
    void computeDescriptors(const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

private:
    auto calcKeyPoints(torch::Tensor &&score);
    auto removeBorders(torch::Tensor &keypoints, torch::Tensor &scores,
                       const int border, const int height, const int width);
    auto calcDescriptors(torch::Tensor kpts, torch::Tensor &&descs);

private:
    std::shared_ptr<torch::jit::script::Module> module_;
    torch::Device device_ = torch::Device(torch::kCPU);
    double keypoint_threshold_;
    int remove_borders_;
    int max_keypoints_;
    torch::Tensor scores_;
    torch::Tensor descriptors_;
};

#endif
