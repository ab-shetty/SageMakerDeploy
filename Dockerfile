FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-gpu-py310-cu118-ubuntu20.04

ENV PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/opt/conda/bin:$PATH"

RUN apt-get update && apt-get install -y \
    libgl1 \
    ffmpeg \
    g++ \
    ninja-build

RUN python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
#RUN python -c "import sagemaker; print(sagemaker.__version__)"
RUN python -c "import pkg_resources; print(pkg_resources.get_distribution('sagemaker-huggingface-inference-toolkit').version)"

# Install compatible version of flash-attn for PyTorch 2.1.0
RUN pip install flash-attn==2.3.6 --no-build-isolation

RUN pip install transformers==4.41.2 accelerate==1.0.1
RUN pip install decord ffmpeg-python imageio opencv-python
RUN pip install --upgrade sagemaker sagemaker-huggingface-inference-toolkit

# RUN python -c "import sagemaker; print(sagemaker.__version__)"
RUN python -c "import pkg_resources; print(pkg_resources.get_distribution('sagemaker-huggingface-inference-toolkit').version)"
