reader() {
    TARGET="/private/home/jgu/data/3d_ssl2/ScannetScan/data_render_test"   # data destination (change here)
    filename=$1
    frame_skip=25

    scene=$(basename -- "$filename")
    scene="${scene%.*}"
    echo "Find sens data: $filename $scene"
    python -u reader.py --filename $filename --output_path $TARGET/$scene --frame_skip $frame_skip --export_depth_images --export_poses --export_intrinsics
    echo "Extract point-cloud data"
    python -u point_cloud_extractor.py --input_path $TARGET/$scene --output_path $TARGET/$scene/pcd --save_npz
    echo "Compute partial scan overlapping"
    python -u compute_full_overlapping.py --input_path $TARGET/$scene/pcd
}
export -f reader


parallel -j 50 --linebuffer time reader ::: `find  /datasets01/scannet/082518/scans/scene*/*.sens`
# parallel -j 1 --linebuffer time reader ::: `find  /datasets01/scannet/082518/scans/scene0024_00/*.sens`
