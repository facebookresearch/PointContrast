import glob, os
import numpy as np
import cv2
import argparse

from plyfile import PlyData, PlyElement

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--input_path', required=True, help='path to sens file to read')
parser.add_argument('--output_path', required=True, help='path to output folder')
parser.add_argument('--save_npz', action='store_true')
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.output_path):
    os.mkdir(opt.output_path)

def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

# Load Depth Camera Intrinsic
depth_intrinsic = np.loadtxt(opt.input_path + '/intrinsic/intrinsic_depth.txt')
print('Depth intrinsic: ')
print(depth_intrinsic)

# Compute Camrea Distance (just for demo, so you can choose the camera distance in frame sampling)
poses = sorted(glob.glob(opt.input_path + '/pose/*.txt'), key=lambda a: int(os.path.basename(a).split('.')[0]))
depths = sorted(glob.glob(opt.input_path + '/depth/*.png'), key=lambda a: int(os.path.basename(a).split('.')[0]))

# # Get Aligned Point Clouds.
for ind, (pose, depth) in enumerate(zip(poses, depths)):
    name = os.path.basename(pose).split('.')[0]
    print('='*50, ': {}'.format(pose))
    depth_img = cv2.imread(depth, -1) # read 16bit grayscale image
    pose = np.loadtxt(poses[ind])
    print('Camera pose: ')
    print(pose)
    
    depth_shift = 1000.0
    x,y = np.meshgrid(np.linspace(0,depth_img.shape[1]-1,depth_img.shape[1]), np.linspace(0,depth_img.shape[0]-1,depth_img.shape[0]))
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:,:,0] = x
    uv_depth[:,:,1] = y
    uv_depth[:,:,2] = depth_img/depth_shift
    uv_depth = np.reshape(uv_depth, [-1,3])
    uv_depth = uv_depth[np.where(uv_depth[:,2]!=0),:].squeeze()
    
    intrinsic_inv = np.linalg.inv(depth_intrinsic)
    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx = depth_intrinsic[0,3]
    by = depth_intrinsic[1,3]
    point_list = []
    n = uv_depth.shape[0]
    points = np.ones((n,4))
    X = (uv_depth[:,0]-cx)*uv_depth[:,2]/fx + bx
    Y = (uv_depth[:,1]-cy)*uv_depth[:,2]/fy + by
    points[:,0] = X
    points[:,1] = Y
    points[:,2] = uv_depth[:,2]
    points_world = np.dot(points, np.transpose(pose))
    print(points_world.shape)

    if opt.save_npz:
        print('Saving npz file...')
        np.savez(opt.output_path + '/{}.npz'.format(name), pcd=points_world[:, :3])
    else:
        print('Saving ply file...')
        write_ply(points_world, opt.output_path + '/{}.ply'.format(name))
