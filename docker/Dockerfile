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

# The base image ships an NVIDIA CUDA apt repo whose signing key is expired/rotated;
# drop it since none of the packages below come from it.
RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# The base image's conda env ships Python 3.7, but av2 (installed below from
# requirement.txt) requires Python>=3.8 (per docs/install.md). `conda install
# python=3.8` fails to solve on this Ubuntu 18.04 base (glibc 2.27 vs. the
# 2.28 conda-forge builds now need), so use a plain venv with deadsnakes
# instead and reinstall torch against it before installing anything else.
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.8 python3.8-venv python3.8-distutils python3.8-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN python3.8 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install MMCV, MMDetection and MMSegmentation (versions per MapTRv2 docs/install.md)
RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
RUN pip install mmdet==2.14.0
RUN pip install mmsegmentation==0.14.1
# Unpinned `timm` pulls in the latest release, which needs torch.fx APIs not
# present in torch 1.9.1; pin to a version contemporaneous with this codebase.
RUN pip install timm==0.6.13

RUN conda clean --all

# Copy the MapTR working tree into the image. Build with the repo root as the
# context, e.g. `docker build -f docker/Dockerfile -t maptrv2 .`
# MapTR bundles its own customized mmdetection3d fork (v0.17.2).
COPY . /MapTR
WORKDIR /MapTR

# Install the bundled mmdetection3d (v0.17.2 fork) and the GKT CUDA op.
# Pre-install its unpinned runtime deps via pip (wheels) first: `setup.py develop`
# falls back to legacy easy_install for unmet deps, and current scikit-image no
# longer ships an easy_install-compatible sdist.
RUN pip install --no-cache-dir -r /MapTR/mmdetection3d/requirements/runtime.txt
RUN cd /MapTR/mmdetection3d && python setup.py develop
RUN cd /MapTR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn && python setup.py build install

# Install remaining MapTR requirements (shapely, av2, ...).
RUN pip install --no-cache-dir -r /MapTR/requirement.txt

# nuScenes data prep / training needs the devkit (not listed in requirement.txt).
RUN pip install --no-cache-dir nuscenes-devkit

# The base image ships numpy 1.19.5, but av2 imports numpy.typing (added in
# 1.20) at module load time. Pin below 1.24 to keep the np.float/np.int
# aliases mmdet 2.14.0/mmcv-full 1.4.0 still use (removed in numpy 1.24).
RUN pip install --no-cache-dir "numpy==1.23.5"

# mmcv-full/mmdet/nuscenes-devkit/av2 all transitively pull in non-headless
# opencv-python, which links its GUI backend against libGL.so.1. On
# Singularity/Apptainer clusters with --nv, the host's newer libGL.so.1 gets
# bind-mounted over the container's own and fails to load against this
# image's older glibc (GLIBC_ABI_DT_RELR, added in glibc 2.36, missing here).
# Swap to the headless build — same OpenCV, no GL/X11 linkage at all — so
# cv2 never touches libGL.so.1 regardless of what the host injects.
RUN OPENCV_VERSION=$(pip show opencv-python | sed -n 's/^Version: //p') \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir "opencv-python-headless==${OPENCV_VERSION}"
