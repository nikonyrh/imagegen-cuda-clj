FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf \
    build-essential \
    curl \
    cython \
    gettext \
    jq \
    libtool-bin \
    nano \
    pkg-config \
    procps \
  && rm -rf /var/lib/apt/lists/*

# Update package managers
RUN pip install --upgrade pip \
  && conda update -y conda \
  && conda update -y python

# Install packages
COPY requirements.txt /requirements.txt

RUN conda install \
    -c defaults \
    -c conda-forge \
    -y $(cat requirements.txt | grep '# conda' | sed -r 's/\s*#.*$//' | tr '\n' ' ') \
  && conda clean -y --all

RUN pip install --no-cache-dir \
  $(cat requirements.txt | grep '# pip' | sed -r -e 's/\s*#.*$//' -e 's/([^<>=])=/\1==/' | tr '\n' ' ')

ENTRYPOINT ["python"]

