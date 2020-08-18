#include "superglue.h"
#include "spdlog/spdlog.h"

SuperGlue::SuperGlue(const YAML::Node &glue_config) : image_rows_(glue_config["image_rows"].as<int>()),
                                                      image_cols_(glue_config["image_cols"].as<int>())
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

    module_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(workspace + "../superglue/models/model/SuperGlue.pt", device_));
    assert(module_ != nullptr);
    spdlog::info("Load model successful!");
}

void SuperGlue::match(std::vector<cv::KeyPoint> &kpts0, std::vector<cv::KeyPoint> &kpts1, cv::Mat &desc0, cv::Mat &desc1)
{
    cv::Mat kpts_mat0(kpts0.size(), 2, CV_32F); // [n_keypoints, 2]  (y, x)
    cv::Mat kpts_mat1(kpts1.size(), 2, CV_32F); // [n_keypoints, 2]  (y, x)
    cv::Mat scores_mat0(kpts0.size(), 1, CV_32F);
    cv::Mat scores_mat1(kpts1.size(), 1, CV_32F);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < kpts0.size(); ++i)
    {
        kpts_mat0.at<float>(i, 0) = static_cast<float>(kpts0[i].pt.y);
        kpts_mat0.at<float>(i, 1) = static_cast<float>(kpts0[i].pt.x);
        scores_mat0.at<float>(i) = static_cast<float>(kpts0[i].response);
    }
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < kpts1.size(); ++i)
    {
        kpts_mat1.at<float>(i, 0) = static_cast<float>(kpts1[i].pt.y);
        kpts_mat1.at<float>(i, 1) = static_cast<float>(kpts1[i].pt.x);
        scores_mat1.at<float>(i) = static_cast<float>(kpts1[i].response);
    }

    auto kpts0_tensor = torch::from_blob(kpts_mat0.data, {static_cast<long>(kpts0.size()), 2}, torch::kFloat).to(device_);
    auto kpts1_tensor = torch::from_blob(kpts_mat1.data, {static_cast<long>(kpts1.size()), 2}, torch::kFloat).to(device_);
    auto scores0_tensor = torch::from_blob(scores_mat0.data, {static_cast<long>(kpts0.size())}, torch::kFloat).to(device_);
    auto scores1_tensor = torch::from_blob(scores_mat1.data, {static_cast<long>(kpts1.size())}, torch::kFloat).to(device_);
#ifdef DEBUG
    std::cout << "kpts0_tensor:" << kpts0_tensor.sizes() << std::endl;
    std::cout << "kpts1_tensor:" << kpts1_tensor.sizes() << std::endl;
    std::cout << "scores0_tensor:" << scores0_tensor.sizes() << std::endl;
    std::cout << "scores1_tensor:" << scores1_tensor.sizes() << std::endl;
#endif
    auto a = normalizeKeypoints(kpts0_tensor);
}

torch::Tensor SuperGlue::normalizeKeypoints(torch::Tensor &kpts)
{
    auto one = torch::ones(1).to(kpts);
    std::cout << one.sizes() << std::endl;
    auto size = torch::stack({one * image_cols_, one * image_rows_});
    std::cout << size.sizes() << std::endl;
    std::cout << size << std::endl;
    auto center = size / 2;
    auto scaling = std::get<1>(size.max(1, true)) * 0.7;
    std::cout << std::get<0>(size.max(1, true)) << std::endl;
    std::cout << std::get<1>(size.max(1, true)) << std::endl;
    // kpts - center
}