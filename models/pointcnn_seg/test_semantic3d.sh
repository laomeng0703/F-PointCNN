#!/usr/bin/env bash

gpu=
setting=
ckpt=
repeat=
save_ply=

usage() { echo "test pointcnn_seg with -g gpu_id -x setting -l ckpt -r repeat -s options"; }

gpu_flag=0
setting_flag=0
ckpt_flag=0
repeat_flag=0
while getopts g:x:l:r:sh opt; do
  case $opt in
  g)
    gpu_flag=1;
    gpu=$(($OPTARG))
    ;;
  x)
    setting_flag=1;
    setting=${OPTARG}
    ;;
  l)
    ckpt_flag=1;
    ckpt=${OPTARG}
    ;;
  r)
    repeat_flag=1;
    repeat=$(($OPTARG))
    ;;
  s)
    save_ply=-s
    ;;
  h)
    usage; exit;;
  esac
done

shift $((OPTIND-1))

if [ $gpu_flag -eq 0 ]
then
  echo "-g option is not presented!"
  usage; exit;
fi

if [ $setting_flag -eq 0 ]
then
  echo "-x option is not presented!"
  usage; exit;
fi

if [ $ckpt_flag -eq 0 ]
then
  echo "-l option is not presented!"
  usage; exit;
fi

if [ $repeat_flag -eq 0 ]
then
  echo "-r option is not presented!"
  usage; exit;
fi

echo "Test setting $setting on GPU $gpu with checkpoint $ckpt! with repeat $repeat"
CUDA_VISIBLE_DEVICES=$gpu python3 ../test_general_seg.py -t ../../data/semantic3d/out_part/test_data_files.txt -f ../../data/semantic3d/out_part/test_data  -l $ckpt -m pointcnn_seg -x $setting -r $repeat $save_ply
