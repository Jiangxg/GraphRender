#ifndef DISPLAY_H
#define DISPLAY_H
#include "aabb/include/intersect_gpu.h"

class RenderLoader{
    public:
        unsigned char* dev_input_alpha, *dev_input_k0, *dev_output;
        float* dev_input_patches, *dev_input_reference_camera, *dev_input_target_camera_ex, *dev_input_target_camera_in_inv;
        int* dev_input_patches_no, *dev_input_information;
        int width, height, channels;
        int patch_num, n_max;
        float depthMin, depthMax;

        RenderLoader();
        ~RenderLoader();
};

void readModel(std::string path, RenderLoader &PMPI, int out_width, int out_height);

void display(RenderLoader &PMPI,
  int out_width, 
  int out_height,
  int x, int y);

void retrieveMemory(unsigned char* &out_data, unsigned char* dev_output, int out_width, int out_height);

#endif