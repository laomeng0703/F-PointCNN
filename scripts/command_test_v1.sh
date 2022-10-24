#/bin/bash
python train/test.py --gpu 0 --num_point 2048 --model pointcnn --model_path train/log_v16/model.ckpt --output train/detection_results_v16 --data_path kitti/frustum_carpedcyc_val_rgb_detection.pickle --from_rgb_detection --idx_path kitti/image_sets/val.txt --from_rgb_detection
train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ train/detection_results_v16
