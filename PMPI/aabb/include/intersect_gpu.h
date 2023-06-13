#ifndef INTERSECT_GPU_H
#define INTERSECT_GPU_H
void aabb_intersect_point_kernel_wrapper(int* dev_input_information, int* dev_input_patches_no, unsigned char* dev_input_alpha, unsigned char* dev_input_k0, float* dev_input_reference_camera,
               float* dev_input_patches, float* dev_input_target_camera_ex, float* dev_input_target_camera_in_inv, unsigned char* dev_output, int out_width, int out_height, int patch_num, int n_max, float depthMin, float depthMax);

#endif