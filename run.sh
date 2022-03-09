export CUDA_VISIBLE_DEVICES=0 #windows和Mac下不需要执行该命令
python3.6 tools/eval.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_roadsign.pdparams
#python3.6 tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml
#python3.6 tools/infer.py -c configs/ppyolo/ppyolo.yml -o use_gpu=true weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams --infer_dir=dataset/roadsign_voc/images/ 1>logoutput.txt 2>logoutput1.txt
