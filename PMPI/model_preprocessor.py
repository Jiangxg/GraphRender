import numpy as np
import sys

ROWS = 33
COLS = 40

#PLANES = 32

OFFSET = 216
SIZE_PATCH = 36

HEIGHT = 1008
WIDTH = 756

def main():
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    input_path_start = model_path + r'\depth_origin.txt'
    input_path_end = model_path + r'\depth_final.txt'
    input_path_intrinsics = data_path + r'\cameras.txt'
    output_patches = model_path + r'\patches.txt'
    output_reference_camera = model_path + r'\reference_camera.txt'
    output_information = model_path + r'\information.txt'
    output_patches_no = model_path + r'\patches_depth_no.txt'
    PLANES = int(sys.argv[3])
    print("PLANES:{}".format(PLANES))

    with open(input_path_start, 'r') as f:
        depth_start = f.readlines()
    depth_start  = [float(x.strip()) for x in depth_start if x.strip()]
    depth_start = np.array(depth_start)
    depth_start = depth_start.reshape((ROWS, COLS))


    with open(input_path_end, 'r') as f:
        depth_end = f.readlines()
    depth_end  = [float(x.strip()) for x in depth_end if x.strip()]
    depth_end = np.array(depth_end)
    depth_end = depth_end.reshape((ROWS, COLS))


    # read intrinsics
    with open(input_path_intrinsics, 'r') as f:
        lines = f.readlines()
    camera_params = lines[3]

    # data: [3985 2988 3260.53 3260.53 1992.5 1494]
    data = camera_params.split()
    print("camera parameters: {}".format(data))
    width_in = float(data[2])
    height_in = float(data[3])
    fx = float(data[4]) * WIDTH / width_in
    fy = float(data[5]) * HEIGHT / height_in
    px = float(data[6]) * WIDTH / width_in
    py = float(data[7]) * HEIGHT / height_in
    
    intrinsics = np.array( [[fx, 0, px],
                      [0,   fy,     py],
                      [0,   0,      1]])
    
    #save intrinsics of reference camera
    np.savetxt(output_reference_camera, intrinsics.reshape((1, -1)), fmt='%0.8f')

    intrinsics_inverse = np.linalg.inv(intrinsics)

    # construct depths : [PLANES, ROWS*COLS] depth
    depths = np.dstack([np.linspace(depth_start[i,j], depth_end[i,j], num=PLANES) for i in range(ROWS) for j in range(COLS)])
    depths = np.squeeze(depths)

    # PMPI_center: [ROWS, COLS, 3], xp, yp, 1
    PMPI_center = np.empty((ROWS, COLS, 3))
    PMPI_x_start = SIZE_PATCH/2 - OFFSET
    PMPI_y_start = SIZE_PATCH/2 - OFFSET
    PMPI_x_end = SIZE_PATCH* COLS - SIZE_PATCH/2 - OFFSET
    PMPI_y_end = SIZE_PATCH* ROWS - SIZE_PATCH/2 - OFFSET
    PMPI_center[..., 0] = np.repeat(np.linspace(PMPI_x_start, PMPI_x_end, COLS)[None], ROWS, axis=0)
    PMPI_center[..., 1] = np.repeat(np.linspace(PMPI_y_start, PMPI_y_end, ROWS)[...,None], COLS, axis=1)
    PMPI_center[..., 2] = 1

    # Patches: [PLANES, ROWS*COLS, 3]
    Patches = np.matmul(intrinsics_inverse, np.expand_dims(PMPI_center, axis=3))
    Patches = np.repeat(np.expand_dims(Patches, axis=0), PLANES, axis=0).reshape(PLANES, ROWS*COLS, 3)
    
    Patches[..., 0] = (Patches[..., 0] / Patches[..., 2]) * depths
    Patches[..., 1] = (Patches[..., 1] / Patches[..., 2]) * depths
    Patches[..., 2] = depths

    # Patches: [PLANES*ROWS*COLS, 3]
    Patches = Patches.reshape((-1, 3))
    
    # calculate l_half
    point_x1 = np.matmul(intrinsics_inverse, np.array([SIZE_PATCH/2, 0, 1]).reshape((3,1)))
    point_x2 = np.matmul(intrinsics_inverse, np.array([0, 0, 1]).reshape((3,1)))
    point_x1 = point_x1 / point_x1[2]
    point_x2 = point_x2 / point_x2[2]
    
    # Patches_half: [PLANES*ROWS*COLS, 1]
    Patches_half = ((point_x1[0] - point_x2[0]) * depths).reshape((-1, 1))
    
    # Patches: [PLANES*ROWS*COLS, 4], xp, yp, dp, l_half
    Patches = np.append(Patches, Patches_half, axis=1)
    

    # construct Patches_no: [PLANES*ROWS*COLS, 1]
    Patches_no = np.empty((PLANES, ROWS*COLS, 1))
    for i in range(PLANES):
        Patches_no[i, :, 0] = i + 1
    Patches_no = Patches_no.reshape((-1, 1))
    # sort Patches_no:
    Patches_no = Patches_no[np.argsort(Patches[:, 2])]

    # save Patches_no
    np.savetxt(output_patches_no, Patches_no, fmt='%d')

    # sort Patches in the order of front-to-back dp
    Patches = Patches[np.argsort(Patches[:, 2])]



    # save patches.txt
    np.savetxt(output_patches, Patches, fmt='%0.8f')
    
    # save information.txt
    information = np.array([PLANES, SIZE_PATCH, HEIGHT, WIDTH, OFFSET, OFFSET]).reshape((1, -1))
    np.savetxt(output_information, information, fmt='%d')

if __name__ == '__main__':
    main()