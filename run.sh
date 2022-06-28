#export CUDA_VISIBLE_DEVICES=2 #windows和Mac下不需要执行该命令
# export FLAGS_enable_eager_mode=1
# CUDA_VISIBLE_DEVICES=2 python3.7 tools/eval.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_roadsign.pdparams > newprofiler_infer.txt;
#CUAD_VISIBLE_DEVICES=2 nsys profile -c cudaProfilerApi python3.7 tools/eval.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_roadsign.pdparams
#for((i=0;i<10;i++));
#do
CUDA_VISIBLE_DEVICES=2 python3.7 tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml > profiler0621.txt;
#  CUDA_VISIBLE_DEVICES=3 FLAGS_host_trace_level=2  python3.7 tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml > test_statistic_data_with_nsys/newprofiler_testflag.txt;
# CUDA_VISIBLE_DEVICES=3 nsys profile -c cudaProfilerApi python3.7 tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml > test_statistic_data_with_nsys/nsys.txt;
#python3.7 -m paddle.distributed.launch --gpus 2,3 tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml > test_statistic_data_with_nsys/newprofiler_distributed.txt;

#done
# CUDA_VISIBLE_DEVICES=2 python3.7 tools/infer.py -c configs/ppyolo/ppyolo.yml -o use_gpu=true weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams --infer_dir=dataset/roadsign_voc/images/
#CUAD_VISIBLE_DEVICES=2 nsys profile -c cudaProfilerApi  python3.7 tools/infer.py -c configs/ppyolo/ppyolo.yml -o use_gpu=true weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams --infer_dir=dataset/roadsign_voc/images/ #1>logoutput.txt 2>logoutput1.txt
# CUDA_VISIBLE_DEVICES=3 python3.7 tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml --output_dir=./inference_model
#CUDA_VISIBLE_DEVICES=3 python3.7 tools/export_model.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml --output_dir=./inference_model
