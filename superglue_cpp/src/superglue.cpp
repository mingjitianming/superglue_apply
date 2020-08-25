#include "superglue.h"
#include "spdlog/spdlog.h"

SuperGlue::SuperGlue(const YAML::Node &glue_config) : image_width_(glue_config["image_width"].as<int>()),
                                                      image_height_(glue_config["image_height"].as<int>()),
                                                      sinkhorn_iterations_(glue_config["sinkhorn_iterations"].as<int>()),
                                                      match_threshold_(glue_config["match_threshold"].as<float>()),
                                                      weight_(glue_config["weight"].as<std::string>())

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
    assert(weight_ == "outdoor" || weight_ == "indoor");
    if (weight_ == "indoor")
    {
        module_ = std::make_shared<torch::jit::script::Module>(
            torch::jit::load(
                workspace +
                    "../superglue/models/model/SuperGlue_indoor.pt",
                device_));
        spdlog::info("Loaded SuperGlue model ('indoor' weights)");
    }

    else
    {
        module_ = std::make_shared<torch::jit::script::Module>(
            torch::jit::load(
                workspace +
                    "../superglue/models/model/SuperGlue_outdoor.pt",
                device_));
        spdlog::info("Loaded SuperGlue model ('outdoor' weights)");
    }
    assert(module_ != nullptr);
    spdlog::info("Load model successful!");
}

auto SuperGlue::logSinkhornIterations(torch::Tensor &Z, torch::Tensor &log_mu, torch::Tensor &log_nu, const int iters)
{
    auto u = torch::zeros_like(log_mu);
    auto v = torch::zeros_like(log_nu);
    for (auto i = 0; i < iters; ++i)
    {
        u = log_mu - torch::logsumexp(Z + v.unsqueeze(1), 2);
        v = log_nu - torch::logsumexp(Z + u.unsqueeze(2), 1);
    }
    // std::cout << u << std::endl;
    // std::cout << v << std::endl;
    return Z + u.unsqueeze(2) + v.unsqueeze(1);
}

auto SuperGlue::logOptimalTransport(const torch::Tensor &&scores, torch::Tensor &&alpha, const int iters)
{
    int b = scores.size(0);
    int m = scores.size(1);
    int n = scores.size(2);
    auto one = torch::ones(1).squeeze();
    auto ms = (m * one).to(scores);
    auto ns = (n * one).to(scores);

    auto bins0 = alpha.expand({b, m, 1});
    auto bins1 = alpha.expand({b, 1, n});
    // std::cout << scores << std::endl;
    alpha = alpha.expand({b, 1, 1});
    auto couplings = torch::cat({torch::cat({scores, bins0}, -1),
                                 torch::cat({bins1, alpha}, -1)},
                                1);
    auto norm = -(ms + ns).log();
    auto log_mu = torch::cat({norm.expand(m), ns.log().unsqueeze_(0) + norm});
    auto log_nu = torch::cat({norm.expand(n), ms.log().unsqueeze_(0) + norm});
    log_mu = log_mu.unsqueeze(0).expand({b, -1});
    log_nu = log_nu.unsqueeze(0).expand({b, -1});
    // std::cout << couplings << std::endl;
    auto Z = logSinkhornIterations(couplings, log_mu, log_nu, iters);
    Z = Z - norm;
    return Z;
}

auto SuperGlue::arangeLike(const torch::Tensor &x, const int dim)
{
    auto a = torch::arange(x.size(dim));
    return a;
}

std::tuple<std::vector<int>, std::vector<float>, std::vector<int>, std::vector<float>>
SuperGlue::match(std::vector<cv::KeyPoint> &kpts0, std::vector<cv::KeyPoint> &kpts1, cv::Mat &desc0, cv::Mat &desc1)
{
    cv::Mat kpts_mat0(kpts0.size(), 2, CV_32F); // [n_keypoints, 2]  (y, x)
    cv::Mat kpts_mat1(kpts1.size(), 2, CV_32F); // [n_keypoints, 2]  (y, x)
    cv::Mat scores_mat0(kpts0.size(), 1, CV_32F);
    cv::Mat scores_mat1(kpts1.size(), 1, CV_32F);
    // std::cout << desc0.at<float>(0, 0) << std::endl;
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < kpts0.size(); ++i)
    {
        kpts_mat0.at<float>(i, 0) = static_cast<float>(kpts0[i].pt.x);
        kpts_mat0.at<float>(i, 1) = static_cast<float>(kpts0[i].pt.y);
        scores_mat0.at<float>(i) = static_cast<float>(kpts0[i].response);
    }
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < kpts1.size(); ++i)
    {
        kpts_mat1.at<float>(i, 0) = static_cast<float>(kpts1[i].pt.x);
        kpts_mat1.at<float>(i, 1) = static_cast<float>(kpts1[i].pt.y);
        scores_mat1.at<float>(i) = static_cast<float>(kpts1[i].response);
    }

    auto kpts0_tensor = torch::from_blob(kpts_mat0.data, {static_cast<long>(kpts0.size()), 2}, torch::kFloat).to(device_);
    auto kpts1_tensor = torch::from_blob(kpts_mat1.data, {static_cast<long>(kpts1.size()), 2}, torch::kFloat).to(device_);
    auto scores0_tensor = torch::from_blob(scores_mat0.data, {1, static_cast<long>(kpts0.size())}, torch::kFloat).to(device_);
    auto scores1_tensor = torch::from_blob(scores_mat1.data, {1, static_cast<long>(kpts1.size())}, torch::kFloat).to(device_);
    auto descriptors0 = torch::from_blob(desc0.data, {1, desc0.cols, desc0.rows}, torch::kFloat).to(device_);
    auto descriptors1 = torch::from_blob(desc1.data, {1, desc1.cols, desc1.rows}, torch::kFloat).to(device_);
    auto kpts0_t = normalizeKeypoints(kpts0_tensor);
    auto kpts1_t = normalizeKeypoints(kpts1_tensor);

    // std::cout << descriptors0[0][0][0] << std::endl;
    // std::cout << descriptors0[0][0][15] << std::endl;
    // std::cout << descriptors1[0][0][0] << std::endl;
    // std::cout << descriptors1[0][0][15] << std::endl;

    // #ifdef DEBUG
    //     std::cout << "kpts0_tensor:" << kpts0_tensor.sizes() << std::endl;
    //     std::cout << "kpts1_tensor:" << kpts1_tensor.sizes() << std::endl;
    //     std::cout << "scores0_tensor:" << scores0_tensor.sizes() << std::endl;
    //     std::cout << "scores1_tensor:" << scores1_tensor.sizes() << std::endl;
    //     std::cout << "descriptors0:" << descriptors0.sizes() << std::endl;
    //     std::cout << "descriptors1:" << descriptors1.sizes() << std::endl;

    //     std::cout << "kpts0_t:" << kpts0_t.sizes() << std::endl;
    //     std::cout << "kpts1_t:" << kpts1_t.sizes() << std::endl;
    // #endif

    torch::Dict<std::string, torch::Tensor> data;
    data.insert("keypoints0", kpts0_t);
    data.insert("keypoints1", kpts1_t);
    data.insert("scores0", scores0_tensor);
    data.insert("scores1", scores1_tensor);
    data.insert("descriptors0", scores0_tensor);
    data.insert("descriptors1", scores1_tensor);
#ifdef DEBUG
    auto start = std::chrono::steady_clock::now();
#endif
    auto out = module_->forward({data}).toTuple();
    auto scores = logOptimalTransport(out->elements()[0].toTensor(), out->elements()[1].toTensor(), sinkhorn_iterations_);
    // auto scores = out->elements()[0].toTensor();
#ifdef DEBUG
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "SuperpGlue module elapsed time: " << elapsed_seconds.count() << "s\n";
#endif
    std::cout << scores[0][0][0] << std::endl;
    std::cout << scores[0][0][1] << std::endl;
    auto [values0, indices0] = scores.slice(1, 0, scores[0].size(0) - 1).slice(2, 0, scores[0].size(1) - 1).max(2);
    auto [values1, indices1] = scores.slice(1, 0, scores[0].size(0) - 1).slice(2, 0, scores[0].size(1) - 1).max(1);

    auto mutual0 = torch::arange(indices0.size(1)).unsqueeze(0).to(device_) == indices1.gather(1, indices0);
    auto mutual1 = torch::arange(indices1.size(1)).unsqueeze(0).to(device_) == indices0.gather(1, indices1);

    auto zero = torch::zeros(1).squeeze().to(device_);
    auto mscores0 = torch::where(mutual0, values0.exp(), zero);
    auto mscores1 = torch::where(mutual1, mscores0.gather(1, indices1), zero);

    auto valid0 = mutual0 & (mscores0 > match_threshold_);
    auto valid1 = mutual1 & valid0.gather(1, indices1);

    indices0 = torch::where(valid0, indices0, torch::full_like(indices0, -1, torch::kInt64).to(device_));
    indices1 = torch::where(valid1, indices1, torch::full_like(indices1, -1, torch::kInt64).to(device_));

    // std::cout << indices0.sizes() << std::endl;
    // std::cout << mscores0.sizes() << std::endl;
    // std::cout << indices1.sizes() << std::endl;
    // std::cout << mscores1.sizes() << std::endl;

    std::vector<int> key_indices0;
    std::vector<float> point_scores0;
    std::vector<int> key_indices1;
    std::vector<float> point_scores1;
    key_indices0.reserve(indices0.size(1));
    key_indices0.reserve(indices1.size(1));
    point_scores0.reserve(mscores0.size(1));
    point_scores1.reserve(mscores1.size(1));

    for (int i = 0; i < indices0.size(1); ++i)
    {
        key_indices0.emplace_back(indices0[0][i].item().toInt());
        point_scores0.emplace_back(mscores0[0][i].item().toFloat());
    }

    for (int i = 0; i < indices1.size(1); ++i)
    {
        key_indices1.emplace_back(indices1[0][i].item().toInt());
        point_scores1.emplace_back(mscores1[0][i].item().toFloat());
    }
    return std::make_tuple(key_indices0, point_scores0, key_indices1, point_scores1);
}

torch::Tensor SuperGlue::normalizeKeypoints(torch::Tensor &kpts)
{
    auto one = torch::tensor(1).to(kpts);
    // auto one = torch::ones(1).to(kpts);

    auto size = torch::stack({one * image_width_, one * image_height_}, 0)
                    .unsqueeze(0);

    auto center = size / 2;
    auto scaling = std::get<0>(size.max(1, true)) * 0.7;

    // #ifdef DEBUG
    //     std::cout << one.sizes() << std::endl;
    //     std::cout << size.sizes() << std::endl;
    //     std::cout << size << std::endl;
    // std::cout << std::get<0>(size.max(1, true)) << std::endl;
    // std::cout << std::get<1>(size.max(1, true)) << std::endl;
    // #endif
    return ((kpts - center) / scaling).unsqueeze_(0);
}