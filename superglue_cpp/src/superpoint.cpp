#include "superpoint.h"
#include "spdlog/spdlog.h"
#include <chrono>

SuperPoint::SuperPoint(const YAML::Node &config_node) : keypoint_threshold_(config_node["keypoint_threshold"].as<double>()),
                                                        remove_borders_(config_node["remove_borders"].as<int>())
{
    if (torch::cuda::is_available())
    {
        device_ = torch::Device(torch::kCUDA);
        spdlog::info("CUDA is available!");
    }
    else
    {
        spdlog::warn("CUDA is not available!");
    }

    module_ = std::make_shared<torch::jit::script::Module>(torch::jit::load("/home/zmy/project_ws/superglue_apply/superglue/models/model/SuperPoint.pt", device_));
    assert(module_ != nullptr);
    spdlog::info("Load model successful!");
}

SuperPoint::~SuperPoint()
{
}

void SuperPoint::detect(const cv::Mat &image)
{
    torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 1}, torch::kByte);
    img_tensor = img_tensor.permute({0, 3, 1, 2}).to(device_);
    img_tensor = img_tensor.to(torch::kFloat) / 255;
#ifdef DEBUG
    auto start = std::chrono::steady_clock::now();
#endif
    auto out = module_->forward({img_tensor}).toTuple();

#ifdef DEBUG
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
#endif
    auto keypoints = keyPoints(std::move(out->elements()[0].toTensor()));
    torch::Tensor temp = out->elements()[0].toTensor();
}

torch::Tensor SuperPoint::keyPoints(torch::Tensor &&score)
{
    torch::Tensor keypoints;
#ifdef DEBUG
    std::cout << score.sizes() << std::endl;
#endif
    auto keypts = torch::nonzero(score[0] > keypoint_threshold_);
    std::cout << keypts.sizes() << std::endl;
    auto kpts = keypts.t();
    // std::cout << "keypts:" << keypts.sizes() << std::endl;
    // std::cout << "kpts:" << kpts.sizes() << std::endl;
    // std::cout << "score:" << score.sizes() << std::endl;
    // std::cout << "index:" << score[0].index(kpts[0]).sizes() << std::endl;
    auto keypts_score = score[0].index({kpts[0], kpts[1]});
    removeBorders(keypts, keypts_score, remove_borders_, score.size(0), score.size(1));

    return torch::randint(1, 9, c10::IntArrayRef({2, 2}));
}

std::pair<torch::Tensor, torch::Tensor> SuperPoint::removeBorders(torch::Tensor &keypoints, torch::Tensor &scores, int border, int height, int width)
{
    auto mask_h = (keypoints.slice(1, 0, 1) >= border) & (keypoints.slice(1, 0, 1) < (height - border));
    auto mask_w = (keypoints.slice(1, 1, 2) >= border) & (keypoints.slice(1, 1, 2) < (width - border));
    auto mask = mask_h & mask_w;
    std::cout << mask.sizes() << std::endl;
    return std::make_pair(keypoints.index(mask.squeeze()), scores.index(mask.squeeze()));
}
