ARG PYTORCH="1.9.1"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"

# Avoid apt hanging on the interactive tzdata prompt.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV, MMDetection and MMSegmentation (versions per MapTRv2 docs/install.md)
RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
RUN pip install mmdet==2.14.0
RUN pip install mmsegmentation==0.14.1
RUN pip install timm

RUN conda clean --all

# Copy the MapTR working tree into the image. Build with the repo root as the
# context, e.g. `docker build -f docker/Dockerfile -t maptrv2 .`
# MapTR bundles its own customized mmdetection3d fork (v0.17.2).
COPY . /MapTR
WORKDIR /MapTR

# Install the bundled mmdetection3d (v0.17.2 fork) and the GKT CUDA op.
RUN cd /MapTR/mmdetection3d && python setup.py develop
RUN cd /MapTR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn && python setup.py build install

# Install remaining MapTR requirements (shapely, av2, ...).
RUN pip install --no-cache-dir -r /MapTR/requirement.txt

# nuScenes data prep / training needs the devkit (not listed in requirement.txt).
RUN pip install --no-cache-dir nuscenes-devkit
