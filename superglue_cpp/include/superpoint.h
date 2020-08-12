#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <memory>
#include <opencv2/core/core.hpp>
#include <yaml-cpp/yaml.h>

class SuperPoint
{

public:
    SuperPoint(const YAML::Node& config_node);
    ~SuperPoint();
    void detect(const cv::Mat &image);

private:
    torch::Tensor keyPoints();
    std::shared_ptr<torch::jit::script::Module> module;
    torch::Device device = torch::Device(torch::kCPU);
};
