Preprocessing ScanNet Dataset
====
1. Request downloading the ScanNet dataset from https://github.com/ScanNet/ScanNet and unzip to ``SCANNET_PATH``.
2. Extract scene data and construct the pre-training corpus to ``TARGET``. 


Following is an example to extract the training data for every 25 frames. Using *50* processes, it takes around *15* hours to fully extract and preprocess the datasets.
```bash
export TARGET=<path_to_target_data> 
export SCANNET=<path_to_downloaded_data>
export FRAME_SKIP=25
export JOBS=50

reader() {
    filename=$1

    scene=$(basename -- "$filename")
    scene="${scene%.*}"
    echo "Find sens data: $filename $scene"
    python -u reader.py --filename $filename --output_path $TARGET/$scene --frame_skip $FRAME_SKIP --export_depth_images --export_poses --export_intrinsics
    echo "Extract point-cloud data"
    python -u point_cloud_extractor.py --input_path $TARGET/$scene --output_path $TARGET/$scene/pcd --save_npz
    echo "Compute partial scan overlapping"
    python -u compute_full_overlapping.py --input_path $TARGET/$scene/pcd
}
export -f reader


parallel -j $JOBS --linebuffer time reader ::: `find  $SCANNET/scans/scene*/*.sens`
```
