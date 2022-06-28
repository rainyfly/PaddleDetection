import os
import tempfile
import numpy as np
import paddle
import paddle.utils as utils
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
from paddle.fluid import compiler, Program, program_guard
import paddle.profiler as profiler

paddle.enable_static()

def build_program(compile_program=True):
      startup_program = fluid.Program()
      main_program = fluid.Program()
      with fluid.program_guard(main_program, startup_program):
          image = fluid.layers.data(name='x', shape=[784], dtype='float32')
          hidden1 = fluid.layers.fc(input=image, size=64, act='relu')
          i = layers.zeros(shape=[1], dtype='int64')
          counter = fluid.layers.zeros(shape=[1],
                                        dtype='int64',
                                        force_cpu=True)
          until = layers.fill_constant([1], dtype='int64', value=10)
          data_arr = layers.array_write(hidden1, i)
          cond = fluid.layers.less_than(x=counter, y=until)
          while_op = fluid.layers.While(cond=cond)
          with while_op.block():
              hidden_n = fluid.layers.fc(input=hidden1, size=64, act='relu')
              layers.array_write(hidden_n, i, data_arr)
              fluid.layers.increment(x=counter, value=1, in_place=True)
              layers.less_than(x=counter, y=until, cond=cond)

          hidden_n = layers.array_read(data_arr, i)
          hidden2 = fluid.layers.fc(input=hidden_n, size=64, act='relu')
          predict = fluid.layers.fc(input=hidden2, size=10, act='softmax')
          label = fluid.layers.data(name='y', shape=[1], dtype='int64')
          cost = fluid.layers.cross_entropy(input=predict, label=label)
          avg_cost = fluid.layers.mean(cost)
          batch_size = fluid.layers.create_tensor(dtype='int64')
          batch_acc = fluid.layers.accuracy(input=predict,
                                            label=label,
                                            total=batch_size)

      optimizer = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
      opts = optimizer.minimize(avg_cost, startup_program=startup_program)

      if compile_program:
          # TODO(luotao): profiler tool may have bug with multi-thread parallel executor.
          # https://github.com/PaddlePaddle/Paddle/pull/25200#issuecomment-650483092
          exec_strategy = fluid.ExecutionStrategy()
          exec_strategy.num_threads = 1
          train_program = fluid.compiler.CompiledProgram(
              main_program).with_data_parallel(loss_name=avg_cost.name,
                                                exec_strategy=exec_strategy)
      else:
          train_program = main_program
      return train_program, startup_program, avg_cost, batch_size, batch_acc


def run_iter(exe, main_program, fetch_list):
    x = np.random.random((32, 784)).astype("float32")
    y = np.random.randint(0, 10, (32, 1)).astype("int64")
    outs = exe.run(main_program,
                    feed={
                        'x': x,
                        'y': y
                    },
                    fetch_list=fetch_list)

def net_profiler( exe,
                  state,
                  tracer_option,
                  batch_range=None,
                  use_parallel_executor=False,
                  use_new_api=False):
    main_program, startup_program, avg_cost, batch_size, batch_acc = build_program(
        compile_program=use_parallel_executor)
    exe.run(startup_program)

    with profiler.Profiler() as prof:
        for iter in range(10):
            run_iter(exe, main_program,
                          [avg_cost, batch_acc, batch_size])
            print(iter)
            prof.step()

def test_cpu_profiler():
    exe = fluid.Executor(fluid.CPUPlace())
    net_profiler(exe,'CPU',"Default")

test_cpu_profiler()