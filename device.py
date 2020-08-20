"""
Abstraction for device in p2p training
"""
from abc import ABC
import os
from models.lenet import SimpleConvNet
import torch


class DeviceBase(ABC):
  """
  Device abstract base class
  """

  def __init__(self, **kwargs):
    raise NotImplementedError('Device not implemented')


class Device(DeviceBase):
  def __init__(self, rank, dataset):
    self.rank = rank
    self.model = SimpleConvNet()
    self.dataset = dataset
    print('Device {} started'.format(rank))

  def get_device_details(self):
    print('Device Process id {}'.format(os.getpid()))
    print('Parent Process id {}'.format(os.getppid()))

  def run(self):
    return torch.rand(size=(2, 4))