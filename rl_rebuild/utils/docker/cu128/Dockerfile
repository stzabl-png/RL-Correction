FROM sharpadev/sharpawavesdk:5.0.0.8b7e

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-dev \
    python3-venv \
    libevdev-dev

RUN pip install --break-system-packages \
    gymnasium==1.2.1 \
    omegaconf==2.3.0 \
    pynput \
    termcolor \
    tensorboardX

RUN pip install --break-system-packages --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0

RUN pip cache purge

RUN apt-get remove -y \
    build-essential \
    libevdev-dev

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
