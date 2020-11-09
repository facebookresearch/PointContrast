Preprocessing ScanNet Pair Dataset
====
1. Request downloading the ScanNet dataset from https://github.com/ScanNet/ScanNet and unzip to ``SCANNET_DIR``.
2. Extract scene data and construct the pre-training corpus to ``TARGET_DIR``. 


Following is an example to extract the training data for every 25 frames.

```bash
export TARGET_DIR=<path_to_target_data> 
export SCANNET_DIR=<path_to_downloaded_data>
export FRAME_SKIP=25
export JOBS=50

reader() {
    filename=$1

    scene=$(basename -- "$filename")
    scene="${scene%.*}"
    echo "Find sens data: $filename $scene"
    python -u reader.py --filename $filename --output_path $TARGET_DIR/$scene --frame_skip $FRAME_SKIP --export_depth_images --export_poses --export_intrinsics
    echo "Extract point-cloud data"
    python -u point_cloud_extractor.py --input_path $TARGET_DIR/$scene --output_path $TARGET_DIR/$scene/pcd --save_npz
    echo "Compute partial scan overlapping"
    python -u compute_full_overlapping.py --input_path $TARGET_DIR/$scene/pcd
}
export -f reader


parallel -j $JOBS --linebuffer time reader ::: `find  $SCANNET_DIR/scans/scene*/*.sens`
```

Then generate the dataset list file (filtering out pairs with less than 30% overlap) which will be used in the PointContrast code:
```
python generate_list.py --target_dir $TARGET_DIR
```

### Notes 

 The full data generation process will sample 843K pairs of point cloud (you can download the full ``example_dataset/overlp-30-full.txt`` list [here](https://www.dropbox.com/s/vqvrmg0umve364n/overlap-30-full.txt?dl=0) for reference). Using *50* processes, it takes around *15* hours to fully extract and preprocess the datasets and could use up to 1TB of disk space. 

For debugging purpose, we provide a 50-pair *example* dataset that can be downloaded from [here](https://www.dropbox.com/s/9ppm0s4veow0yst/data_f25.tar?dl=0). Please extract it to ``example_dataset/`` after the download. The pair list for training is provided at ``example_dataset/overlap-30-50p-subset.txt``. This will help you walk through the training process though the results will not be useful.

Given limited resources, we recommend subsmapling the scannet dataset before sampling the pairs. In our experience, 20K pairs shoud be good enough to provide good pretraining performance.