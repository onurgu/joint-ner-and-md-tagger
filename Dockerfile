#FROM ubuntu:16.04
FROM python:3.5

# Build dependencies
#RUN echo "deb-src http://archive.ubuntu.com/ubuntu/ xenial main restricted" >> /etc/apt/sources.list && apt-get update && apt-get install -y python-pip git mercurial build-essential cmake libopenblas-dev

#RUN apt-get update && apt-get install -y python-pip git mercurial build-essential cmake libopenblas-dev

#RUN apt-get update && apt-get -y upgrade

#RUN apt-get install curl

#RUN curl -O https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py

#RUN pip install --upgrade "pip < 21.0"
RUN pip install wheel
#
RUN pip install cython

RUN pip install dynet

#RUN pip install numpy==1.12.0
#
#ENV EIGEN_VERSION 699b659
## ENV EIGEN_VERSION 346ecdb
##RUN cd /opt && git clone --depth 1 https://gitlab.com/libeigen/eigen
##        hg clone https://bitbucket.org/eigen/eigen/ -r ${EIGEN_VERSION}
#COPY eigen /opt/eigen
#
## code for installing Intel MKL. You need to supply l_mkl_2018.1.163.tgz file
##COPY l_mkl_2018.1.163.tgz /opt/
##COPY docker/my_silent.cfg /opt/l_mkl_2018.1.163/
##RUN cd /opt/ && \
##    tar zxvf l_mkl_2018.1.163.tgz && cd /opt/l_mkl_2018.1.163 && \
##    ./install.sh --silent "my_silent.cfg"
##ENV LD_PRELOAD /opt/intel/mkl/lib/intel64/libmkl_def.so:/opt/intel/mkl/lib/intel64/libmkl_avx2.so:/opt/intel/mkl/lib/intel64/libmkl_core.so:/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so:/opt/intel/mkl/lib/intel64/libmkl_intel_thread.so:/opt/intel/lib/intel64_lin/libiomp5.so
#
## DyNet, version 4234759
#ENV DYNET_VERSION 2.0.1
#RUN cd /opt && \
#        git clone https://github.com/clab/dynet.git && \
#        cd dynet && \
#        git checkout ${DYNET_VERSION} && \
#        mkdir build && \
#        cd build && \
#        cmake .. -DEIGEN3_INCLUDE_DIR=/opt/eigen -DPYTHON=`which python` && \
#        make -j2 && \
#        cd python && \
#        python ../../setup.py build --build-dir=.. --skip-build install
#
#RUN cd /opt/dynet/build && make -j2 install
#
## if you want to use Intel MKL, change the above cmake line to
## cmake .. -DEIGEN3_INCLUDE_DIR=/opt/eigen -DPYTHON=`which python` -DMKL_ROOT /opt/intel/mkl && \
#
#ENV DYLD_LIBRARY_PATH /opt/dynet/build/dynet/
#ENV LD_LIBRARY_PATH /opt/dynet/build/dynet/

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
