FROM python:3.8

USER root

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \  
    libglib2.0-0 \
    libxrender1 \
    libfontconfig1 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p app
WORKDIR /app

RUN git clone https://github.com/flyxiv/nino.git 
RUN cd nino && python -m venv venv && . ./venv/bin/activate && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117 && pip install mmcv -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.1/index.html && deactivate
RUN git clone https://github.com/flyxiv/mmdetection.git
ENV PYTHONPATH=/app/mmdetection/:$PYTHONPATH

CMD ["bash"]