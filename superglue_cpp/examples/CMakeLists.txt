add_executable(superpoint_test
                superpoint_test.cpp)
target_link_libraries(superpoint_test
                     superpoint
)

add_executable(superglue_test
               superglue_test.cpp)

target_link_libraries(superglue_test
                     superpoint
                     superglue
)