"""run.py:"""
# !/usr/bin/env python
# !/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from dataset import make_data_partition
from args import get_args
from device import Device

args = get_args()



def run(rank, size, data):
  """ Distributed function to be implemented later. """
  group = dist.new_group([x for x in range(size)])
  d = Device(rank, data)
  tensor = d.run()
  print('Before all reduce')
  print('Device ', rank, ' has data ', tensor[0])
  dist.all_reduce(tensor, op=dist.reduce_op.SUM, group=group)
  print('After all reduce')
  print('Device ', rank, ' has data ', tensor[0])
  print('---------------')

def init_process(rank, size, fn, data, backend='gloo'):
  """ Initialize the distributed environment. """
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '29500'
  dist.init_process_group(backend, rank=rank, world_size=size)
  fn(rank, size, data)


if __name__ == "__main__":
  size = 3
  processes = []
  datasets =  make_data_partition(args)
  for rank in range(size):
    p = Process(target=init_process, args=(rank, size, run, datasets[rank+1]))
    p.start()
    processes.append(p)

  for p in processes:
    p.join()
