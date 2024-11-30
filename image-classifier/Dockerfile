ARG PYTHON_VERSION=3.12
ARG DEBIAN_OS=slim-bookworm


FROM --platform=${TARGETPLATFORM:-linux/amd64} ghcr.io/openfaas/of-watchdog:0.10.4 AS watchdog
FROM --platform=${TARGETPLATFORM:-linux/amd64} python:${PYTHON_VERSION}-${DEBIAN_OS} AS build

COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog

ARG ADDITIONAL_PACKAGE
# Alternatively use ADD https:// (which will not be cached by Docker builder)
ARG UPGRADE_PACKAGES=false

RUN apt-get update -qy \
    && if [ "${UPGRADE_PACKAGES}" = "true" ] || [ "${UPGRADE_PACKAGES}" = "1" ]; then apt-get upgrade -qy; fi \
    && apt-get install -qy --no-install-recommends gcc make ${ADDITIONAL_PACKAGE} \
    && apt-get install curl -y \
    && rm -rf /var/lib/apt/lists/*

ENV HOME /home/app
ENV PATH=$HOME/conda/bin:$PATH

RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && curl -sSL https://github.com/openfaas-incubator/of-watchdog/releases/download/0.5.1/of-watchdog > /usr/bin/fwatchdog \
    && chmod 777 /tmp/miniconda.sh 


# Add non root user
RUN addgroup --system app \
    && adduser app --system --ingroup app  --home /home/app \
    && chown app:app /home/app

USER app

ENV PATH=$PATH:/home/app/.local/bin
#install conda
RUN bash /tmp/miniconda.sh -bfp $HOME/conda \
    && conda install -y python=3 \
    && conda install -y pytorch torchvision torchaudio cpuonly -c pytorch \
    && conda update conda \
    && conda clean --all --yes 
    

WORKDIR /home/app/

COPY --chown=app:app index.py           .
COPY --chown=app:app requirements.txt   .
USER root
RUN pip install --no-cache-dir -r requirements.txt
RUN rm -rf /tmp/miniconda.sh
USER app

RUN mkdir -p function
RUN touch ./function/__init__.py
WORKDIR /home/app/function/
COPY --chown=app:app function/requirements.txt	.
RUN pip install --no-cache-dir --user -r requirements.txt

USER root
COPY --chown=app:app function/   .

FROM build AS test

ARG TEST_COMMAND=tox
ARG TEST_ENABLED=true
RUN [ "$TEST_ENABLED" = "false" ] && echo "skipping tests" || eval "$TEST_COMMAND"


FROM build AS ship
WORKDIR /home/app/

USER app

# Set up of-watchdog for HTTP mode
ENV fprocess="python index.py"
ENV cgi_headers="true"
ENV mode="http"
ENV upstream_url="http://127.0.0.1:5000"

HEALTHCHECK --interval=5s CMD [ -e /tmp/.lock ] || exit 1

CMD ["fwatchdog"]
