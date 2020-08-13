#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <memory>
#include <opencv2/core/core.hpp>
#include <yaml-cpp/yaml.h>

class SuperPoint
{

public:
    SuperPoint(const YAML::Node &config_node);
    ~SuperPoint();
    void detect(const cv::Mat &image);

private:
    torch::Tensor keyPoints(torch::Tensor &&score);
    std::pair<torch::Tensor, torch::Tensor>
         removeBorders(torch::Tensor &keypoints, torch::Tensor &scores,
                                                          int border, int height, int width);

private:
    std::shared_ptr<torch::jit::script::Module> module_;
    torch::Device device_ = torch::Device(torch::kCPU);
    double keypoint_threshold_;
    int remove_borders_;
};
