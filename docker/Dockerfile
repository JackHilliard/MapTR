ARG PYTORCH="2.1.0"
ARG CUDA="11.8"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# 8.6 covers Ampere (e.g. RTX 3070/3090); 9.0+PTX covers Hopper (H100) and,
# via PTX forward-compat JIT, anything newer released after this image (e.g.
# Blackwell RTX 50-series).
ENV TORCH_CUDA_ARCH_LIST="8.6 9.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"

# Avoid apt hanging on the interactive tzdata prompt.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV, MMDetection and MMSegmentation. Bumped from the versions in
# docs/install.md (mmcv-full==1.4.0/mmdet==2.14.0/mmsegmentation==0.14.1,
# built for CUDA 11.1) to support Hopper (H100) GPUs, which need CUDA>=11.8 —
# the first CUDA release with real sm_90 support. mmdet/mmsegmentation are
# bumped alongside mmcv because both hard-assert an mmcv upper bound in their
# own __init__.py that the old pins don't cover (mmdet<=1.4.0,
# mmsegmentation<=1.4.0); 2.28.2/0.30.0 are the last mmdet 2.x/mmsegmentation
# 0.x releases, with mmcv upper bounds of 1.8.0, comfortably covering 1.7.2.
#
# mmcv-full MUST be built from source here, not installed from OpenMMLab's
# prebuilt wheel index. Their prebuilt cu118/torch2.1.0 wheel was compiled
# with OpenMMLab's own (unknown, non-configurable) TORCH_CUDA_ARCH_LIST,
# which does not include Hopper (sm_90) and has no +PTX fallback baked in --
# this is invisible on Ampere (works fine on this sandbox's RTX 3070) and
# only surfaces on an actual H100 as `RuntimeError: CUDA error: no kernel
# image is available for execution on the device` inside mmcv's own CUDA
# ops (e.g. ms_deform_attn, sigmoid_focal_loss) the first time they run.
# Building from source makes mmcv's setup.py pick up this Dockerfile's own
# TORCH_CUDA_ARCH_LIST (set above, "8.6 9.0+PTX"), same as every other
# custom op in this image (GKT, mmdetection3d's ops) already does.
RUN MMCV_WITH_OPS=1 pip install --no-cache-dir --no-binary mmcv-full "mmcv-full==1.7.2"
# mmcv 1.7.2's single-GPU MMDataParallel path (mmcv.parallel.Scatter.forward)
# calls PyTorch's private torch.nn.parallel._functions._get_stream() with a
# raw int device id; torch>=2.x's version of that function requires a
# torch.device object instead (`AttributeError: 'int' object has no
# attribute 'type'`). Patch the installed file to wrap the device id.
RUN sed -i \
    "s/streams = \[_get_stream(device) for device in target_gpus\]/streams = [_get_stream(torch.device('cuda', device) if isinstance(device, int) else device) for device in target_gpus]/" \
    /opt/conda/lib/python3.10/site-packages/mmcv/parallel/_functions.py
RUN pip install mmdet==2.28.2
RUN pip install mmsegmentation==0.30.0
# Unpinned `timm` pulls in the latest release, which needs torch.fx APIs not
# present in older torch; pin to a version contemporaneous with this codebase.
RUN pip install timm==0.6.13

# spconv2 (maintained fork) replaces mmdetection3d's vendored spconv 1.x,
# whose CUDA kernels are incompatible with CUDA 11.8/sm_86+90 (crash with
# "cuda execution failed with error 2" even on trivial inputs) -- see
# mmdetection3d/mmdet3d/ops/spconv/__init__.py, which re-exports this
# instead of the vendored implementation.
RUN pip install --no-cache-dir spconv-cu118==2.3.8

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

# av2 imports numpy.typing (added in numpy 1.20) at module load time. Pin
# below 1.24 to keep the np.float/np.int aliases mmdet/mmcv-full may still
# use (removed in numpy 1.24).
RUN pip install --no-cache-dir "numpy==1.23.5"

# tools/maptrv2/webviewer.py is designed to run outside the container
# (lightweight, no GPU/ML deps needed) -- these let it also run from
# inside, if that's more convenient in a given deployment. matplotlib and
# tensorboard are already present transitively; only flask/seaborn are
# actually missing (seaborn is a dependency of webviewer.py's reuse of
# tools/analysis_tools/analyze_logs.py's log parser, not used directly).
RUN pip install --no-cache-dir flask seaborn

# requirement.txt pins shapely==1.8.5.post1, but nuscenes-devkit (installed
# after it, with an unpinned shapely dependency) silently upgrades it to
# 2.x. shapely 2.0 changed STRtree.query() to return integer indices
# instead of geometry objects -- a breaking API change that crashes MapTR's
# own chamfer-distance eval code (map_utils/tpfp_chamfer.py assumes the
# old object-returning behavior). Re-pin as the last install so it sticks.
RUN pip install --no-cache-dir "shapely==1.8.5.post1"

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
