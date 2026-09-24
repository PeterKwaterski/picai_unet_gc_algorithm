FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -U pip && \
    python3 -m pip install -r /tmp/requirements.txt

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

COPY weights/ /opt/algorithm/weights/
COPY process.py /opt/algorithm/
RUN chown -R algorithm:algorithm /opt/algorithm

USER algorithm

WORKDIR /opt/algorithm

ENTRYPOINT ["python3", "/opt/algorithm/process.py"]

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=picai_unet_baseline_processor
