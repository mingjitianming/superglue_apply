#include "superpoint.h"
#include "spdlog/spdlog.h"
#include <chrono>

SuperPoint::SuperPoint(const YAML::Node &config_node) : keypoint_threshold_(config_node["keypoint_threshold"].as<double>()),
                                                        remove_borders_(config_node["remove_borders"].as<int>()),
                                                        max_keypoints_(config_node["max_keypoints"].as<int>())
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

auto SuperPoint::removeBorders(torch::Tensor &keypoints, torch::Tensor &scores, const int border, const int height, const int width)
{
    auto mask_h = (keypoints.slice(1, 0, 1) >= border) & (keypoints.slice(1, 0, 1) < (height - border));
    auto mask_w = (keypoints.slice(1, 1, 2) >= border) & (keypoints.slice(1, 1, 2) < (width - border));
    auto mask = mask_h & mask_w;
    // std::cout << keypoints.sizes() << std::endl;
    // std::cout << scores.sizes() << std::endl;
    // std::cout << mask << std::endl;
    // std::cout << torch::nonzero(mask.squeeze()) << std::endl;
    // std::cout << keypoints.index(mask.squeeze().t()).sizes() << std::endl;
    // std::cout << scores.index(mask.squeeze().t()).sizes() << std::endl;

    return std::make_pair(keypoints.index(mask.squeeze()), scores.index(mask.squeeze()));
}

auto SuperPoint::calcKeyPoints(torch::Tensor &&score)
{
    auto keypts = torch::nonzero(score[0] > keypoint_threshold_);
    auto kpts = keypts.t();
    auto keypts_score = score[0].index({kpts[0], kpts[1]});

    // auto kpts_and_scores = removeBorders(keypts, keypts_score, remove_borders_, score[0].size(0), score[0].size(1));
    // if (max_keypoints_ > 0 && kpts_and_scores.first.size(0) > max_keypoints_)
    // {
    //     auto scores_and_indices = torch::topk(kpts_and_scores.second, max_keypoints_, 0);
    //     scores = std::get<0>(scores_and_indices);
    //     keypoints = kpts_and_scores.first.index(std::get<1>(scores_and_indices));
    // }

    auto [keypoints, scores] = removeBorders(keypts, keypts_score, remove_borders_, score[0].size(0), score[0].size(1));
    if (max_keypoints_ > 0 && keypoints.size(0) > max_keypoints_)
    {
        auto [ss, indices] = torch::topk(scores, max_keypoints_, 0);
        scores = ss;
        keypoints = keypoints.index(indices);
    }
    keypoints = torch::flip(keypoints, {1});
    return std::make_pair(keypoints, scores);
}

auto SuperPoint::calcDescriptors(torch::Tensor kpts, torch::Tensor &&descs)
{
    int s = 8;
    int b = descs.size(0);
    int h = descs.size(2);
    int w = descs.size(3);

    kpts = kpts - s / 2 + 0.5;
    kpts /= torch::tensor({(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)}).to(kpts);
    kpts = (kpts * 2 - 1).unsqueeze_(0);
    auto descriptors = torch::nn::functional::grid_sample(
        descs, kpts.view({b, 1, -1, 2}),
        torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true));

    return descriptors;
}

std::pair<std::vector<cv::KeyPoint>, cv::Mat> SuperPoint::detect(const cv::Mat &image)
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
    auto [keypoints, scores] = calcKeyPoints(std::move(out->elements()[0].toTensor()));
    auto descriptors = calcDescriptors(keypoints, std::move(out->elements()[1].toTensor()));
    std::vector<cv::KeyPoint> kpts(keypoints.size(0));
    for (auto i = 0; i < keypoints.size(0); ++i)
    {
        auto response = scores[i];
        kpts.emplace_back(keypoints[i][0].item().toFloat(), keypoints[i][1].item().toFloat(), 8, -1, scores[i].item().toFloat());
    }
    cv::Mat desc_mat(cv::Size(descriptors.size(1), descriptors.size(0)), CV_32FC1, descriptors.data<float>());

    return std::make_pair(kpts, desc_mat);
}

// TODO:外部调用
void SuperPoint::computeDescriptors(const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    cv::Mat kpt_mat(keypoints.size(), 2, CV_32F); // [n_keypoints, 2]  (y, x)

    for (size_t i = 0; i < keypoints.size(); i++)
    {
        kpt_mat.at<float>(i, 0) = (float)keypoints[i].pt.y;
        kpt_mat.at<float>(i, 1) = (float)keypoints[i].pt.x;
    }
    auto fkpts = torch::from_blob(kpt_mat.data, {keypoints.size(), 2}, torch::kFloat);
}
