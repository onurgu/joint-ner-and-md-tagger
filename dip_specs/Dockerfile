# assumes docker is built from root directory with dockerfile: dip_specs/Dockerfile
# in root of the project run: docker build -t turkish_ner_tagger -f dip_specs/Dockerfile .
# after building: run it with:
# docker run -ti -p 8080:8080 -v `pwd`/dataset:/opt/ner-tagger-dynet/dataset -v `pwd`/models:/opt/ner-tagger-dynet/models turkish_ner_tagger python main.py --command webapp --model_path model-00002714 --model_epoch_path model-epoch-00000026 --port 8080
# it starts webapp on port 8080

FROM python:3.5

RUN pip install wheel

RUN pip install cython

RUN pip install dynet

RUN mkdir /opt/ner-tagger-dynet

WORKDIR /opt/ner-tagger-dynet

COPY *.py /opt/ner-tagger-dynet/
COPY requirements.txt /opt/ner-tagger-dynet/

RUN pip install pbr

RUN pip install -r requirements.txt

RUN mkdir dataset

COPY evaluation/conlleval evaluation/
COPY evaluation/conlleval.py evaluation/
COPY evaluation/conlleval-runner.sh evaluation/
RUN mkdir -p evaluation/temp/eval_logs/

RUN mkdir models/

COPY web /opt/ner-tagger-dynet/web
COPY utils /opt/ner-tagger-dynet/utils
COPY toolkit /opt/ner-tagger-dynet/toolkit
