FROM nvidia/cuda:12.8.1-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive MUJOCO_GL=egl PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev git git-lfs g++ ca-certificates curl \
    libegl1 libgl1 libopengl0 libglib2.0-0 libusb-1.0-0 && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
ENV RCR_PERCEPTION_HOME=/opt/rcr-perception
ARG CUDA_ARCH=8.9+PTX
RUN python3.12 scripts/perception/setup.py --env /opt/perception-env \
    --robot --cuda --cuda-arch "${CUDA_ARCH}"
ENV PATH=/opt/perception-env/bin:$PATH
ENTRYPOINT ["python", "scripts/perception/run.py"]
CMD ["doctor", "--robot"]
