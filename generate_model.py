python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml --output_dir=./inference_model \
 -o weights=weights/yolov3_darknet53_270e_coco.pdparams

# model = ErnieModel.from_pretrained('ernie-2.0-large-en') 
# model.eval()
# # input_spec = [paddle.static.InputSpec(
# #         shape=[1, 60], dtype='int32')]
# input_spec = [
#             paddle.static.InputSpec(
#                 shape=[None, None], dtype="int64"),  # input_ids
#             paddle.static.InputSpec(
#                 shape=[None, None], dtype="int64")  # segment_ids
#         ]

# # paddle.jit.save(model, 'test_add_graph_large_model', input_spec)
# logwriter = LogWriter(logdir='/work/VisualDL/test_add_graph/ernie-2.0-large-en')
# logwriter.add_graph(model, input_spec)

model = AutoModel.from_pretrained('ernie-3.0-medium-zh')
model.eval()
# input_spec = [paddle.static.InputSpec(
#         shape=[1, 60], dtype='int32')]
input_spec = [
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64")  # segment_ids
        ]
logwriter = LogWriter(logdir='/work/VisualDL/test_add_graph/ernie-3.0-medium-zh')
logwriter.add_graph(model, input_spec)

# paddle.jit.save(model, 'test_add_graph_large_model', input_spec)
# logwriter = LogWriter(logdir='/work/VisualDL/test_add_graph/ernie-gram-en')
# 

model = AutoModel.from_pretrained('chinese-electra-small')
model.eval()
# input_spec = [paddle.static.InputSpec(
#         shape=[1, 60], dtype='int32')]
input_spec = [
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64")  # segment_ids
        ]
logwriter = LogWriter(logdir='/work/VisualDL/test_add_graph/chinese-electra-small')
logwriter.add_graph(model, input_spec)