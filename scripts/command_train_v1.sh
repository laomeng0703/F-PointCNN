#/bin/bash
python train/train.py --gpu 0 --model pointcnn --log_dir train/log_v1 --num_point 2048 --max_epoch 21 --batch_size 10 --decay_step 50000 --decay_rate 0.8
