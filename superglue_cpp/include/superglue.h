#ifndef SUPERGLUE_H
#define SUPERGLUE_H

#include "config.h"
#include <memory>
#include <opencv2/core/core.hpp>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

class SuperGlue
{
public:
    SuperGlue(const YAML::Node &glue_config);
    ~SuperGlue() = default;
    void match(std::vector<cv::KeyPoint> &kpts0, std::vector<cv::KeyPoint> &kpts1, cv::Mat &desc0, cv::Mat &desc1);

private:
    /**
     * @brief Normalize keypoints locations based on image image_shape
     * 
     * @param kpts 
     * @return torch::Tensor 
     */
    torch::Tensor normalizeKeypoints(torch::Tensor &kpts);

private:
    std::shared_ptr<torch::jit::script::Module>
        module_;
    torch::Device device_ = torch::Device(torch::kCPU);
    int image_rows_;
    int image_cols_;
    std::string weight_;
};

#endif