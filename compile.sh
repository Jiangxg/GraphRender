nvcc -arch=compute_86 -c PMPI/display.cu
nvcc -arch=compute_86 -c PMPI/aabb/src/intersect_gpu.cu
ar rcs libdisplaydemo.a display.o intersect_gpu.o
qmake -makefile GraphRender.pro
make
rm display.o intersect_gpu.o libdisplaydemo.a
