#include <fstream>
#include "cuda_runtime.h"
// #include "lodepng.h"
// #include "aabb/src/intersect.cu"
#include <string>
#define STB_IMAGE_IMPLEMENTATION
#include "../stb/stb_image.h"
#include <iostream>
#include <sstream>
// #include "opencv2/core.hpp"
// #include "opencv2/imgproc.hpp"
// #include "opencv2/highgui.hpp"
// #include <opencv2/core/utils/filesystem.hpp>
#include <vector>
#include <chrono>
#include "display.h"


#define NUMINTRINSICS 9  // intrinsics 的元素个数
#define FACTOR 2 // 最多的平面数
#define THREADNUM 1024

unsigned char* dev_input_alpha, *dev_input_k0, *dev_output;
float* dev_input_patches, *dev_input_reference_camera, *dev_input_target_camera_ex, *dev_input_target_camera_in_inv;
int* dev_input_patches_no, *dev_input_information;
int read_flag = 0; 

int width, height, channels, alpha_channels;

void readModel(std::string path, RenderLoader &PMPI, int out_width, int out_height){

    //load the patches.txt, reference_camera.txt and patches_depth_no.txt
    // constrcut path of .txt files
    std::vector<float> patches;
    std::string path_patches;
    path_patches = path + std::string("/") + std::string("patches.txt");
    
    // read patches.txt, each line contains (xp, yp, dp, l_half)
    std::stringstream ss;
    std::string line;
    std::ifstream patches_infile(path_patches);
    if (!patches_infile) {
        std::cout << "Failed to open patches.txt" << std::endl;
        exit(1);
    }
    while (std::getline(patches_infile, line)) {
        if (!line.empty()) {
            ss.str(line);
            std::string single;
            //std::vector<float> patch;
            // 按照空格分割
            while (getline(ss, single, ' ')) {
                patches.push_back(atof(single.c_str()));
            }
            ss.clear(); //必须加，不然写不到string里
        }
    }


    // read intrinsics of reference camera from reference_camera.txt
    std::vector<float> intrinsics; // 
    std::string path_reference_camera;
    path_reference_camera = path + std::string("/") + std::string("reference_camera.txt");

    std::ifstream reference_camera_infile(path_reference_camera);
    if (!reference_camera_infile) {
        std::cout << "Failed to open reference_camera.txt" << std::endl;
        exit(1);
    }
    ss.clear();
    while (std::getline(reference_camera_infile, line)) {
        if (!line.empty()) {
            ss.str(line);
            std::string single;
            // 按照空格分割
            while (getline(ss, single, ' ')) {
                intrinsics.push_back(atof(single.c_str()));
            }
            ss.clear();
        }
    }

    // read information.txt
    std::vector<int> information; // 
    std::string path_information;
    path_information = path + std::string("/") + std::string("information.txt");

    std::ifstream information_infile(path_information);
    if (!information_infile) {
        std::cout << "Failed to open information.txt" << std::endl;
        exit(1);
    }
    ss.clear();
    while (std::getline(information_infile, line)) {
        if (!line.empty()) {
            ss.str(line);
            std::string single;
            // 按照空格分割
            while (getline(ss, single, ' ')) {
                information.push_back(atoi(single.c_str()));
            }
            ss.clear();
        }
    }

    // read patches_depth_no.txt
    std::vector<int> patches_no;
    std::string path_patches_no;
    path_patches_no = path + std::string("/") + std::string("patches_depth_no.txt");

    std::ifstream patches_no_infile(path_patches_no);
    if (!patches_no_infile) {
        std::cout << "Failed to open patches_depth_no.txt" << std::endl;
        exit(1);
    }
    ss.clear();
    while (std::getline(patches_no_infile, line)) {
        if (!line.empty()) {
            ss.str(line);
            std::string single;
            // 按照空格分割
            while (getline(ss, single, ' ')) {
                patches_no.push_back(atoi(single.c_str()));
            }
            ss.clear(); //必须加，不然写不到string里
        }
    }

    // read planes.txt
    std::vector<float> planeDepth; // 
    std::string planeDepth_information;
    planeDepth_information = path + std::string("/") + std::string("planes.txt");

    std::ifstream planeDepth_infile(planeDepth_information);
    if (!planeDepth_infile) {
        std::cout << "Failed to open planes.txt" << std::endl;
        exit(1);
    }
    ss.clear();
    while (std::getline(planeDepth_infile, line)) {
        if (!line.empty()) {
            ss.str(line);
            std::string single;
            // 按照空格分割
            while (getline(ss, single, ' ')) {
                planeDepth.push_back(atof(single.c_str()));
            }
            ss.clear();
        }
    }
    PMPI.depthMin = planeDepth[0];
    PMPI.depthMax = planeDepth[1];

    // read alpha and color images to device
    int num_images = information[0];
    int input_width = information[2] + 2 * information[4];
    int input_height = information[3] + 2 * information[5];
    
    // set PMPI.n_max
    //n_max 应该可以由用户自由调整，初始化为2 * planes
    //TODO: 调整
    PMPI.n_max = FACTOR * num_images;

    std::vector<unsigned char> alpha_images;
    std::vector<unsigned char> k0_images;
    

    // TODO: 如果patch_no和patches的数量对不上，报错
    for(int i = 0; i < num_images; i++){

        // get the path of the alpha_image and the image of the current directory
        std::string seq = std::to_string(i);
        std::string path_image_alpha = path + std::string("/") + std::string("alphaImage_") + std::string(4 - seq.length(), '0') + seq + std::string(".png");
        std::string path_out_k0 = path + std::string("/") + std::string("out_k0_") + std::string(4 - seq.length(), '0') + seq + std::string(".png");

        // load the alpha_image and image including their width, height, and channels
        int alpha_channel = 1;
        int k0_channel = 3;
        unsigned char* alpha_temp_char = stbi_load(path_image_alpha.c_str(), &input_width, &input_height, &alpha_channel, 0); 
        unsigned char* image_temp_char = stbi_load(path_out_k0.c_str(), &input_width, &input_height, &k0_channel, 0);
        
        std::vector<unsigned char> image_alpha(alpha_temp_char, alpha_temp_char + input_width * input_height * 1);
        std::vector<unsigned char> out_k0(image_temp_char, image_temp_char + input_width * input_height * 3);

        // append the newly obtained alpha_image and image to the container
        alpha_images.insert(alpha_images.end(), image_alpha.begin(), image_alpha.end());
        k0_images.insert(k0_images.end(), out_k0.begin(), out_k0.end());
    }

    // Move alpha and k0 to device
    // define the size for each space
    size_t alpha_size = input_width * input_height * sizeof(unsigned char) * num_images * 1;
    size_t k0_size = input_width * input_height * sizeof(unsigned char) * num_images * 3;
    size_t output_size = out_width * out_height * sizeof(unsigned char) * 3;

    // allocate space for alpha images and copy the content

    cudaMalloc(&PMPI.dev_input_alpha, alpha_size);
    cudaMemcpy(PMPI.dev_input_alpha, alpha_images.data(), alpha_size, cudaMemcpyHostToDevice);
    
    // allocate space for images and copy the content
    cudaMalloc(&PMPI.dev_input_k0, k0_size);
    cudaMemcpy(PMPI.dev_input_k0, k0_images.data(), k0_size, cudaMemcpyHostToDevice);
 
    // allocate space for rendering output
    cudaMalloc(&PMPI.dev_output, output_size);


    // Move patches, information and reference camera intrisics to device;
    // cpy information to device
    size_t informationSize = information.size() * sizeof(int);
    cudaMalloc(&PMPI.dev_input_information, informationSize);
    cudaMemcpy(PMPI.dev_input_information, information.data(), informationSize, cudaMemcpyHostToDevice);

    // cpy PMPI patches_no to device
    size_t patches_no_size = patches_no.size() * sizeof(int);
    cudaMalloc(&PMPI.dev_input_patches_no, patches_no_size);
    cudaMemcpy(PMPI.dev_input_patches_no, patches_no.data(), patches_no_size, cudaMemcpyHostToDevice);

    // cpy PMPI patches to device
    int num_patch = num_images * (input_width / information[1]) * (input_height / information[1]);
    PMPI.patch_num = num_patch;
    size_t patches_size = num_patch * sizeof(float) * 4;
    cudaMalloc(&PMPI.dev_input_patches, patches_size);
    cudaMemcpy(PMPI.dev_input_patches, patches.data(), patches_size, cudaMemcpyHostToDevice);
    
    //cpy intrinsics to device:
    size_t intrinsics_size = sizeof(float) * NUMINTRINSICS;
    cudaMalloc(&PMPI.dev_input_reference_camera, intrinsics_size);
    cudaMemcpy(PMPI.dev_input_reference_camera, intrinsics.data(), intrinsics_size, cudaMemcpyHostToDevice);
    

    // TODO: 完善内参的控制机制
    // 暂时将其固定在readModel()中
    std::vector<float> target_camera_in_inv = {0.0016166f, 0.f, -0.61109697f, 0.f, 0.00090914f, -0.45820771f, 0.f, 0.f, 1.f};
    size_t camerap_in_size = 9 * sizeof(float);
    cudaMalloc(&PMPI.dev_input_target_camera_in_inv, camerap_in_size);
    cudaMemcpy(PMPI.dev_input_target_camera_in_inv, target_camera_in_inv.data(), camerap_in_size, cudaMemcpyHostToDevice);    

}


// 主函数
void display(
  RenderLoader &PMPI,
  int out_width, 
  int out_height,
  int x, int y
  ){
    //test only, 此段代码的耗时约为13微秒
    //逐渐转换视角, 目标相机内参固定
    std::vector<float> target_camera_ex = {1.f, 0.f, 0.f, -x/10000.f, 0.f, 1.f, 0.f, -y/10000.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
    //std::vector<float> target_camera_ex_temp = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
    size_t camerap_out_size = 16 * sizeof(float);
    cudaMalloc(&PMPI.dev_input_target_camera_ex, camerap_out_size);
    cudaMemcpy(PMPI.dev_input_target_camera_ex, target_camera_ex.data(), camerap_out_size, cudaMemcpyHostToDevice);

    aabb_intersect_point_kernel_wrapper(PMPI.dev_input_information, PMPI.dev_input_patches_no, PMPI.dev_input_alpha, PMPI.dev_input_k0, PMPI.dev_input_reference_camera,
            PMPI.dev_input_patches, PMPI.dev_input_target_camera_ex, PMPI.dev_input_target_camera_in_inv, PMPI.dev_output, out_width, out_height, PMPI.patch_num, PMPI.n_max, PMPI.depthMin, PMPI.depthMax);
    

}

void retrieveMemory(unsigned char* &out_data,
unsigned char* dev_output, 
int out_width, int out_height){
    size_t output_image_size = out_width * out_height * sizeof(unsigned char) * 3;
    cudaMemcpy(out_data, dev_output, output_image_size, cudaMemcpyDeviceToHost);
}

RenderLoader::RenderLoader(){

}

RenderLoader::~RenderLoader(){

}