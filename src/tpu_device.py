import torch_xla.core.xla_model as xm

tpu_device = xm.xla_device()
devices = [tpu_device]
