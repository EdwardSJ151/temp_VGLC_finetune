FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt update && apt install -y build-essential
RUN  pip install "unsloth==2025.3.18" "unsloth_zoo==2025.3.16" jupyterlab
# 4  jupyter-lab --ip=0.0.0.0 --port=8484 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
# 5  history
