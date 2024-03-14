# docker build . -t TripoSR
# docker run --rm -it --net=host --gpus=all --ipc=host -p 7860:7860 TripoSR bash
# in case of WSL2, in gradio_app.py, add `server_name="0.0.0.0"` to `demo.launch()`

FROM nvcr.io/nvidia/pytorch:24.02-py3

WORKDIR /TripoSR
ADD . /TripoSR
RUN pip install -r requirements.txt
RUN pip install gradio
RUN pip uninstall -y opencv-python opencv-python-headless
RUN pip install opencv-python==4.8.0.74

EXPOSE 7860
