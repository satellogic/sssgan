FROM osgeo/gdal
RUN apt update
RUN apt install -y python3-pip 
RUN apt install -y libopencv-dev python3-opencv
RUN pip install tqdm pandas geopandas matplotlib scikit-image
WORKDIR /ds_preparation