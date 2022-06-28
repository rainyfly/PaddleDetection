#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import paddle
from paddle import nn
import paddle.profiler as profiler

# from model import profile


class SingleNet(nn.Layer):
    def __init__(self, in_size, out_size):
        super(SingleNet, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x


def main(args):
    paddle.seed(123)

    model = SingleNet(10, 20)
    model.train()

    x = paddle.randn([10, 10], dtype='float32')
    x.stop_gradient = False

    prof = profiler.Profiler(targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
                             scheduler=[args.profile_start_step, args.profile_end_step],
                             timer_only=not args.prof)
#                             timer_only=True)
    prof.start()
    start = time.time()
    for iter_id in range(args.max_iter):
        if iter_id == 5:
            paddle.device.cuda.synchronize()
            start = time.time()

        # if iter_id >= args.profile_start_step and iter_id < args.profile_end_step:
        #     profile.set_global_profile(args.prof, args.nvprof)
        # profile.switch_profile(args.profile_start_step, args.profile_end_step, 0, iter_id)

        if args.to_static:
            jit_model = paddle.jit.to_static(model)
            y = jit_model(x)
        else:
            y = model(x)
        loss = paddle.mean(y)
        loss.backward()
        prof.step()
        print("[{}/{}] loss: {:.7f}".format(iter_id, args.max_iter, loss.numpy()[0]))

    paddle.device.cuda.synchronize()
    avg_batch_cost = (time.time() - start) / (args.max_iter - 5)
    print("Average batch_cost: {:.5f} ms".format(avg_batch_cost))

    prof.stop()
    prof.summary(op_detail=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iter', default=20, type=int)
    parser.add_argument('--to_static', action='store_true', help='run with jit.to_static.')
    parser.add_argument('--prof', action='store_true', help='run with profiler.')
    parser.add_argument('--nvprof', action='store_true')
    parser.add_argument('--profile_start_step', default=10, type=int)
    parser.add_argument('--profile_end_step', default=20, type=int)
    args = parser.parse_args()

    main(args)