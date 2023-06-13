#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "../include/cuda_utils.h"
#include "../include/cutil_math.h"  // required for float3 vector math

#define THREADNUM 1024
#define ALPHATHRES 0.01f
#define TRANSTHRESH 0.01f

#define PATCHW 40
#define PATCHH 33

int compare(const void *a, const void *b) {
    float *p1 = (float *)a;
    float *p2 = (float *)b;
    if (p1[0] > p2[0]) {
        return 1;
    } else if (p1[0] < p2[0]) {
        return -1;
    } else {
        return 0;
    }
}

__device__ float3 RayAABBIntersection(
  const float3 &ori,
  const float2 &dir,
  const float3 &center,
  float half_voxel) {

  float intersect_x, left_x, right_x, intersect_y, down_y, up_y;

  intersect_x = __fadd_rn(ori.x, __fmul_rn(__fsub_rn(center.z, ori.z), dir.x));
  left_x = __fsub_rn(center.x, half_voxel);
  if (intersect_x < left_x){
    return make_float3(-1.0f, -1.0f, -1.0f);
  }

  right_x = __fadd_rn(center.x, half_voxel);
  if (intersect_x >= right_x){
    return make_float3(-1.0f, -1.0f, -1.0f);
  }

  intersect_y = __fadd_rn(ori.y, __fmul_rn(__fsub_rn(center.z, ori.z), dir.y));
  down_y = __fsub_rn(center.y, half_voxel);
  if (intersect_y < down_y){
    return make_float3(-1.0f, -1.0f, -1.0f);
  }

  up_y = __fadd_rn(center.y, half_voxel);
  if (intersect_y >= up_y){
    return make_float3(-1.0f, -1.0f, -1.0f);
  }

  return make_float3(intersect_x, intersect_y, center.z);
}

// generate ray, search intersections
__global__ void aabb_intersect_point_kernel(
              const int* __restrict__ dev_input_information,
              const int* __restrict__ dev_input_patches_no, 
              const unsigned char* __restrict__ dev_input_alpha, 
              const unsigned char* __restrict__ dev_input_k0, 
              const float* __restrict__ dev_input_reference_camera,
              const float* __restrict__ dev_input_patches, 
              const float* __restrict__ dev_input_target_camera_ex, 
              const float* __restrict__ dev_input_target_camera_in_inv, 
              unsigned char* __restrict__ dev_output, 
              const int out_width, 
              const int out_height,
              const int patch_num,
              const int n_max,
              const float depthMin,
              const float depthMax) {

  // TOOD: 确定__restrict__词的关键作用
  // __restrict__ 关键词对于加速很重要，减少了访问存储器的次数

  // 计算线程的ID(Block和thread都是二维，thread是32x32)
  int Idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 可能存在超过数量的线程
  if (Idx >=  out_width * out_height) return;
  // 计算pixel的图像坐标
  int pixelX = Idx % out_width;
  int pixelY = Idx / out_width;

  float originX = dev_input_target_camera_ex[3];
  float originY = dev_input_target_camera_ex[7];
  float originZ = dev_input_target_camera_ex[11];

  float pointX = dev_input_target_camera_in_inv[0] * pixelX + dev_input_target_camera_in_inv[1] * pixelY + dev_input_target_camera_in_inv[2];
  float pointY = dev_input_target_camera_in_inv[3] * pixelX + dev_input_target_camera_in_inv[4] * pixelY + dev_input_target_camera_in_inv[5];
  float pointZ = dev_input_target_camera_in_inv[6] * pixelX + dev_input_target_camera_in_inv[7] * pixelY + dev_input_target_camera_in_inv[8];

  float wpointX = dev_input_target_camera_ex[0] * pointX + dev_input_target_camera_ex[1] * pointY + dev_input_target_camera_ex[2] * pointZ + dev_input_target_camera_ex[3];
  float wpointY = dev_input_target_camera_ex[4] * pointX + dev_input_target_camera_ex[5] * pointY + dev_input_target_camera_ex[6] * pointZ + dev_input_target_camera_ex[7];
  float wpointZ = dev_input_target_camera_ex[8] * pointX + dev_input_target_camera_ex[9] * pointY + dev_input_target_camera_ex[10] * pointZ + dev_input_target_camera_ex[11];
 
  float dirX = (wpointX - originX) / (wpointZ - originZ);
  float dirY = (wpointY - originY) / (wpointZ - originZ);
  
  
  // sub-frustum culling
  int imageW = dev_input_information[2] + 2 * dev_input_information[4];
  int imageH = dev_input_information[3] + 2 * dev_input_information[5];

  // 33, 40
  int patchW = imageW / dev_input_information[1];
  int patchH = imageH / dev_input_information[1];
  bool hit[PATCHH][PATCHW] = {false};
  
  // TODO: 添加float depthMin, depthMax的支持
  // (xmin, ymin, depth_min) (xmax, ymax, depth_max) in world coordinates
  float xmin = originX + (depthMin - originZ) * dirX;
  float ymin = originY + (depthMin - originZ) * dirY;

  float xmax = originX + (depthMax - originZ) * dirX;
  float ymax = originY + (depthMax - originZ) * dirY;

  // find out patch coordinates of (xmin, ymin, depth_min)
  float patchX1 = (xmin / depthMin) * dev_input_reference_camera[0] + dev_input_reference_camera[2];
  float patchY1 = (ymin / depthMin) * dev_input_reference_camera[4] + dev_input_reference_camera[5];
  int patchXRound1 = ((int)patchX1 + dev_input_information[4]) / dev_input_information[1];
  int patchYRound1 = ((int)patchY1 + dev_input_information[5]) / dev_input_information[1];

  float patchX2 = (xmax / depthMax) * dev_input_reference_camera[0] + dev_input_reference_camera[2];
  float patchY2 = (ymax / depthMax) * dev_input_reference_camera[4] + dev_input_reference_camera[5];
  int patchXRound2 = ((int)patchX2 + dev_input_information[4]) / dev_input_information[1];
  int patchYRound2 = ((int)patchY2 + dev_input_information[5]) / dev_input_information[1];

  
  // from leftupper to righbottom 
  int patchXStart = patchXRound1 > patchXRound2 ? patchXRound2 : patchXRound1;
  int patchXEnd = patchXRound1 > patchXRound2 ? patchXRound1 : patchXRound2;

  int patchYStart = patchYRound1 > patchYRound2 ? patchYRound2 : patchYRound1;
  int patchYEnd = patchYRound1 > patchYRound2 ? patchYRound1 : patchYRound2;

  // square of radius of bounding sphere: size_patch /2 * 1.415(sqrt(2)) ^ 2
  float radSquare = (dev_input_information[1] / 2) * (dev_input_information[1] / 2) * 2;

  // formula line: Ax + By + C = 0
  float A = patchY1 - patchY2;
  float B = patchX2 - patchX1;
  float C = (patchX1 - patchX2) * patchY1 + (patchY2 - patchY1) * patchX1;
  
  
  // 防止光线和patch的相交角度过大时，后续计算sub-frustum和ray的交点有误差， 考虑patch横跨数小于等于四个
  if ((patchXEnd - patchXStart) * (patchYEnd - patchYStart) <= 4) {
    for (int i = patchXStart; i <= patchXEnd; i++) {
      for (int j = patchYStart; j <= patchYEnd; j++) {
        if (i < 0 || i > patchH || j < 0 || j > patchW) continue;
        hit[j][i] = true;
      }
    }
  }

  // sub-frustrum ray intersection test
  for (int i = patchXStart; i <= patchXEnd; i++) {
    for (int j = patchYStart; j <= patchYEnd; j++) {
      if (i < 0 || i > patchH || j < 0 || j > patchW) continue;
      float xt = (i + 0.5) * dev_input_information[1] - dev_input_information[4];
      float yt = (j + 0.5) * dev_input_information[1] - dev_input_information[5];
      float distanceSquare = (A * xt + B * yt + C) * (A * xt + B * yt + C) / (A * A + B * B);
      if (distanceSquare < radSquare) hit[j][i] = true;
    }
  }
  

  int GridXStart, GridXEnd, GridYStart, GridYEnd;
  int GridXStep, GridYStep;
  // compute mode
  // mode1: leftupper to rightbottom
  // mode2: leftbottom to rightupper
  // mode3: rightbottom to leftupper
  // mode4: rightupper to leftdown
  if (patchXRound1 < patchXRound2) {
    GridXStart = 0;
    GridXEnd = PATCHW;
    GridXStep = 1;
  }
  else {
    GridXStart = PATCHW;
    GridXEnd = 0;
    GridXStep = -1;
  }
  
  if (patchYRound1 < patchYRound2) {
    GridYStart = 0;
    GridYEnd = PATCHH;
    GridYStep = 1;
  }
  else {
    GridYStart = PATCHH;
    GridYEnd = 0;
    GridYStep = -1;
  }

  // compute intersections
  int cnt = 0;
  float transparency = 1.f;
  float colorRF = 0.0f;
  float colorGF = 0.0f;
  float colorBF = 0.0f;

  for (int gridx = GridXStart; gridx != GridXEnd; gridx+=GridXStep) {
    for (int gridy = GridYStart; gridy != GridYEnd; gridy+=GridYStep) {
      if (!hit[gridy][gridx]) continue;
      int patchStart = (patch_num / (PATCHH * PATCHW)) * (PATCHW * gridy + gridx);
      int patchEnd = (patch_num / (PATCHH * PATCHW)) * (PATCHW * gridy + gridx + 1);

      for (int k = patchStart; k < patchEnd && cnt < n_max; k++) {


        // 经过测试，如果for循环中只有RayAABBIntersection，也需要80ms
        float3 intersects = RayAABBIntersection(
          make_float3(originX, originY, originZ),
          make_float2(dirX, dirY),
          make_float3(dev_input_patches[k * 4 + 0], dev_input_patches[k * 4 + 1], dev_input_patches[k * 4 + 2]),
          dev_input_patches[k * 4 + 3]);
        
        if (intersects.z > 0.0f){ 
          cnt++;

          float U = (intersects.x / intersects.z) * dev_input_reference_camera[0] + dev_input_reference_camera[2];
          float V = (intersects.y / intersects.z) * dev_input_reference_camera[4] + dev_input_reference_camera[5];
          int planeIdx = dev_input_patches_no[3 * k + 2];

          // 颜色双线性插值
          // TODO: 传入参数in_height, in_width, size_padding
          int imageW = dev_input_information[2] + 2 * dev_input_information[4];
          int imageH = dev_input_information[3] + 2 * dev_input_information[5];
          int leftupperIdx = (planeIdx - 1) * imageW * imageH + ((int)V + dev_input_information[4]) * imageW + ((int)U + dev_input_information[5]);
          int rightupperIdx = (planeIdx - 1) * imageW * imageH + ((int)V + dev_input_information[4]) * imageW + ((int)(U + 1) + dev_input_information[5]);
          int leftbottomIdx = (planeIdx - 1) * imageW * imageH + ((int)(V + 1) + dev_input_information[4]) * imageW + ((int)U + dev_input_information[5]);
          int rightbottomIdx = (planeIdx - 1) * imageW * imageH + ((int)(V + 1) + dev_input_information[4]) * imageW + ((int)(U + 1) + dev_input_information[5]);
          
          //实验显示，将此处到if底部注释掉，耗时~480微秒
          //做颜色插值
          float alpha1 = static_cast<float>(dev_input_alpha[leftupperIdx]);
          float alpha2 = static_cast<float>(dev_input_alpha[rightupperIdx]);
          float alpha3 = static_cast<float>(dev_input_alpha[leftbottomIdx]);
          float alpha4 = static_cast<float>(dev_input_alpha[rightbottomIdx]);

          float weight1 = ((int)(U + 1) - U) * ((int)(V + 1) - V);
          float weight2 = (U - (int)U) * ((int)(V + 1) - V);
          float weight3 = ((int)(U + 1) - U) * (V - (int)V);
          float weight4 = (U - (int)U) * (V - (int)V);

          // alpha插值
          float alphaAll = weight1 * alpha1 + weight2 * alpha2 + weight3 * alpha3 + weight4 * alpha4;

          //实验显示，将此处到if底部注释掉，耗时~480微秒
          // alpha值较大时才考虑
          if (alphaAll < ALPHATHRES) continue;

          //实验显示，将此处到if底部注释掉，耗时~430微秒
          float colorR1 = static_cast<float>(dev_input_k0[leftupperIdx*3]);
          float colorG1 = static_cast<float>(dev_input_k0[leftupperIdx*3 + 1]);
          float colorB1 = static_cast<float>(dev_input_k0[leftupperIdx*3 + 2]);

          float colorR2 = static_cast<float>(dev_input_k0[rightupperIdx*3]);
          float colorG2 = static_cast<float>(dev_input_k0[rightupperIdx*3 + 1]);
          float colorB2 = static_cast<float>(dev_input_k0[rightupperIdx*3 + 2]);

          float colorR3 = static_cast<float>(dev_input_k0[leftbottomIdx*3]);
          float colorG3 = static_cast<float>(dev_input_k0[leftbottomIdx*3 + 1]);
          float colorB3 = static_cast<float>(dev_input_k0[leftbottomIdx*3 + 2]);

          float colorR4 = static_cast<float>(dev_input_k0[rightbottomIdx*3]);
          float colorG4 = static_cast<float>(dev_input_k0[rightbottomIdx*3 + 1]);
          float colorB4 = static_cast<float>(dev_input_k0[rightbottomIdx*3 + 2]);

          //实验显示，将此处到if底部注释掉，耗时~430微秒
          // 颜色插值
          float colorRALL = weight1 * colorR1 + weight2 * colorR2 + weight3 * colorR3 + weight4 * colorR4;
          float colorGALL = weight1 * colorG1 + weight2 * colorG2 + weight3 * colorG3 + weight4 * colorG4;
          float colorBALL = weight1 * colorB1 + weight2 * colorB2 + weight3 * colorB3 + weight4 * colorB4;
          
          // 颜色累积，计算透明度
          //TODO：简化操作
          colorRF = colorRF + colorRALL * (alphaAll / 255.f) * transparency;
          colorGF = colorGF + colorGALL * (alphaAll / 255.f) * transparency;
          colorBF = colorBF + colorBALL * (alphaAll / 255.f) * transparency;
          transparency = transparency * (1.f - alphaAll / 255.f);

          // 早停
          if (transparency < TRANSTHRESH) break;           
        }
      }
    }
  }

  dev_output[Idx * 3] = static_cast<unsigned char>(colorRF);
  dev_output[Idx * 3 + 1] = static_cast<unsigned char>(colorGF);
  dev_output[Idx * 3 + 2] = static_cast<unsigned char>(colorBF);
}


// 仍然运行在CPU+内存
__host__ void aabb_intersect_point_kernel_wrapper(
            int* dev_input_information,
            int* dev_input_patches_no, 
            unsigned char* dev_input_alpha, 
            unsigned char* dev_input_k0, 
            float* dev_input_reference_camera,
            float* dev_input_patches, 
            float* dev_input_target_camera_ex, 
            float* dev_input_target_camera_in_inv, 
            unsigned char* dev_output, 
            int out_width, 
            int out_height,
            int patch_num,
            int n_max,
            float depthMin, 
            float depthMax) {

  // TODO: 调优dim3
  dim3 threadsPerBlock(THREADNUM);
  dim3 numBlocks((out_width * out_height + threadsPerBlock.x - 1) / threadsPerBlock.x);

  // 计时
  cudaEvent_t start, stop;
	float esp_time_gpu;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
  cudaEventRecord(start, 0);// start

  //std::cout << "running aabb_intersect_point_kernel_wrapper" << std::endl;
  //aabb_intersect_point_kernel<<<numBlocks, threadsPerBlock, 3*patch_num*sizeof(int)>>>(dev_input_information, dev_input_patches_no, dev_input_alpha, dev_input_k0, dev_input_reference_camera,
  //            dev_input_patches, dev_input_target_camera_ex, dev_input_target_camera_in_inv, dev_output, out_width, out_height, patch_num, n_max, depthMin, depthMax);

  aabb_intersect_point_kernel<<<numBlocks, threadsPerBlock>>>(dev_input_information, dev_input_patches_no, dev_input_alpha, dev_input_k0, dev_input_reference_camera,
             dev_input_patches, dev_input_target_camera_ex, dev_input_target_camera_in_inv, dev_output, out_width, out_height, patch_num, n_max, depthMin, depthMax);


  cudaEventRecord(stop, 0);// stop
  cudaEventSynchronize(stop);
	cudaEventElapsedTime(&esp_time_gpu, start, stop);
	//printf("Time for the kernel: %f ms\n", esp_time_gpu);
}

