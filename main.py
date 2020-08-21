"""run.py:"""
# !/usr/bin/env python
# !/usr/bin/env python
import os
from math import ceil

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process

from args import get_args
from dataset import get_data
from models.lenet import SimpleConvNet

args = get_args()

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def run(rank, size):
  """ Distributed function to be implemented later. """
  # group = dist.new_group([x for x in range(size)])
  # d = Device(rank, data)
  # tensor = d.run()
  # dist.all_reduce(tensor, op=dist.reduce_op.SUM, group=group)
  torch.manual_seed(1234)
  trainset, testloader = get_data(args)
  model = SimpleConvNet()
  optimizer = optim.SGD(model.parameters(),
                        lr=0.01, momentum=0.5)

  num_batches = ceil(len(trainset.dataset) / float(args.batch_size))
  print(num_batches)
  for epoch in range(10):
    epoch_loss = 0.0
    for data, target in trainset:
      optimizer.zero_grad()
      output = model(data)
      loss = F.nll_loss(output, target)
      epoch_loss += loss.item()
      loss.backward()
      average_gradients(model)
      optimizer.step()
    print('Rank ', dist.get_rank(), ', epoch ',
          epoch, ': ', epoch_loss / num_batches)


def init_process(rank, size, fn, backend='gloo'):
  """ Initialize the distributed environment. """
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '29500'
  dist.init_process_group(backend, rank=rank, world_size=size)
  fn(rank, size)


if __name__ == "__main__":
  size = args.num_devices
  processes = []
  for rank in range(size):
    p = Process(target=init_process, args=(rank, size, run))
    p.start()
    processes.append(p)

  for p in processes:
    p.join()
