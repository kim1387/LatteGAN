FROM gcr.io/tpu-pytorch/xla:nightly_3.8_tpuvm

ENV XRT_TPU_CONFIG="localservice;0;localhost:51011"

WORKDIR /lattegan

RUN mkdir -p /usr/share/tpu/

COPY requirements.txt ./

RUN pip install --upgrade pip \
    pip install -r requirements.txt

RUN pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN apt-get update && apt-get install libgl1 -y
RUN pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.10-cp38-cp38-linux_x86_64.whl
COPY ./run.sh ./

ENTRYPOINT [ "/usr/bin/python", "src/test.py","--yaml_path", "./params/exp085.yaml"]