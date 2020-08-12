#include "superpoint.h"

SuperPoint::SuperPoint(const YAML::Node &config_node)
{   
    if(torch::cuda::is_available())
        device = torch::Device(torch::kCUDA);
    else
        device = torch::Device(torch::kCPU);
    
    
    module = std::make_shared<torch::jit::script::Module>(torch::jit::load("/home/zmy/project_ws/superglue_apply/superglue/models/model/SuperPoint.pt", device));
    assert(module != nullptr);
    std::cout << "Load model successful!" << std::endl;
}

SuperPoint::~SuperPoint()
{
}

void SuperPoint::detect(const cv::Mat &image)
{
    torch::Tensor img_tensor = torch::from_blob(image.data, {1, 1, image.rows, image.cols}, torch::kByte);
    img_tensor = img_tensor.to(torch::kFloat) / 255;
}
