
CUDA_VISIBLE_DEVICES=1, python src/eval.py +experiment=gkt_nuscenes_vehicle_kernel_7x1.yaml \
    data.dataset_dir=/media/ava/DATA3/DATA/sherly/data/nuscenes \
    data.labels_dir=/media/ava/DATA3/DATA/sherly/data/cvt_labels_nuscenes   \
    experiment.ckptt=/media/ava/DATA3/DATA/raj/GKT/segmentation/artifacts/map_segmentation_gkt_7x1_conv_setting2.ckpt  \
    2>&1 | tee log/base2.log