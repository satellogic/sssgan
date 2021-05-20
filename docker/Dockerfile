FROM pure/python:3.6-cuda10.2-cudnn7-runtime
RUN apt-get -y update
RUN apt-get install -y gfortran libopenblas-dev liblapack-dev
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get -y install python3-pip
RUN apt-get install tree
COPY ./requirements.txt .
RUN pip install -r requirements.txt
RUN pip install pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
WORKDIR /SSSGAN
CMD ["/bin/bash"]
