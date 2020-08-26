#ifndef SUPERGLUE_H
#define SUPERGLUE_H

#include "config.h"
#include <memory>
#include <opencv2/core/core.hpp>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <tuple>
#include <yaml-cpp/yaml.h>

class SuperGlue
{
public:
    explicit SuperGlue(const YAML::Node &glue_config);
    ~SuperGlue() = default;
    std::tuple<std::vector<int>, std::vector<float>, std::vector<int>, std::vector<float>>
    match(std::vector<cv::KeyPoint> &kpts0, std::vector<cv::KeyPoint> &kpts1, cv::Mat &desc0, cv::Mat &desc1);

private:
    /**
     * @brief Normalize keypoints locations based on image image_shape
     * 
     * @param kpts 
     * @return torch::Tensor 
     */
    torch::Tensor normalizeKeypoints(torch::Tensor &kpts, int image_width, int image_height);

    /**
     * @brief """ Perform Sinkhorn Normalization in Log-space for stability"""
     * 
     * @param Z 
     * @param log_mu 
     * @param log_nu 
     * @param iters 
     * @return torch::Tensor
     */
    auto logSinkhornIterations(torch::Tensor &Z, torch::Tensor &log_mu, torch::Tensor &log_nu, const int iters);
    /**
     * @brief """ Perform Differentiable Optimal Transport in Log-space for stability"""
     * 
     * @param scores 
     * @param alpha 
     * @param iters 
     * @return torch::Tensor 
     */
    auto logOptimalTransport(const torch::Tensor &&scores, torch::Tensor &&alpha, const int iters);
    auto arangeLike(const torch::Tensor &x, const int dim);

private:
    std::shared_ptr<torch::jit::script::Module>
        module_;
    torch::Device device_ = torch::Device(torch::kCPU);
    int image_width_;
    int image_height_;
    int sinkhorn_iterations_;
    float match_threshold_;
    std::string weight_;
};

#endif