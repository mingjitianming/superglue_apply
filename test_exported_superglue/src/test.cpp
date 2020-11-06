#include "superpoint.h"

#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <yaml-cpp/yaml.h>

int main(int argc, char **argv)
{
    const YAML::Node node = YAML::LoadFile("/home/zmy/project_ws/superglue_apply/superglue_cpp/config/superpoint.yaml");
    SuperPoint sp = SuperPoint(node);
    std::cout << "OK" << std::endl;
    return 0;
}