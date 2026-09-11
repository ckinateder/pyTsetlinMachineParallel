FROM python:3.10.16-slim-bullseye

# dependencies
RUN apt update && apt install -y build-essential git libgl1 libglib2.0-0

# env
ENV OMP_NUM_THREADS=20

WORKDIR /opt/pytsetlin

# Python deps for the example scripts (numpy/pandas/torch/etc. -- see requirements.txt).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build the libTM C extension (Convolutional / MultiClass Tsetlin Machine) into the
# image, so the package is importable out of the box even without a bind mount.
# If you bind-mount your own working copy over this directory for active development
# (`docker run -v $(pwd):$(pwd) ...`, per the README), re-run `pip install -e .` once
# inside the container so the extension gets rebuilt against your live source.
COPY . .
RUN pip install --no-cache-dir -e .
