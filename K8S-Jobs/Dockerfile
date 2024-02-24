FROM jupyter/scipy-notebook

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

#ENV MODEL_DIR=./trainedmodel
#ENV MODEL_FILE=clf.joblib
#ENV METADATA_FILE=metadata.json

COPY train.py ./train.py
COPY inference.py ./inference.py