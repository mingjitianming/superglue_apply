#include "superpoint.h"
#include "spdlog/spdlog.h"
#include <chrono>

SuperPoint::SuperPoint(const YAML::Node &config_node)
{
    if (torch::cuda::is_available())
    {
        device = torch::Device(torch::kCUDA);
        spdlog::info("CUDA is available!");
    }

    module = std::make_shared<torch::jit::script::Module>(torch::jit::load("/home/zmy/project_ws/superglue_apply/superglue/models/model/SuperPoint.pt", device));
    assert(module != nullptr);
    spdlog::info("Load model successful!");
}

SuperPoint::~SuperPoint()
{
}

void SuperPoint::detect(const cv::Mat &image)
{
    torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 1}, torch::kByte);
    img_tensor = img_tensor.permute({0, 3, 1, 2}).to(device);
    img_tensor = img_tensor.to(torch::kFloat) / 255;
#ifdef DEBUG
    auto start = std::chrono::steady_clock::now();
#endif
    auto out = module->forward({img_tensor}).toTuple();
  
#ifdef DEBUG
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
#endif
    std::cout << std::get<0>(*out) << std::endl;
}
