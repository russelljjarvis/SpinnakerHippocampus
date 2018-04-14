FROM russelljarvis/neuronunit
USER jovyan

RUN gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
RUN gpg -a --export E084DAB9 | sudo apt-key add -
RUN sudo apt-get install --fix-missing
RUN sudo apt-get install -f
RUN sudo apt-get autoremove -y

RUN sudo apt update -y
RUN sudo apt upgrade -y

RUN sudo /opt/conda/bin/pip install psutil
ENV QT_QPA_PLATFORM offscreen
RUN pip install dask
RUN pip install distributed
RUN git clone https://github.com/jlizier/jidt
WORKDIR jidt
RUN sudo ant build
RUN sudo apt-get install python-jpype
RUN sudo /opt/conda/bin/pip install JPype1


RUN sudo /opt/conda/bin/pip install git+https://github.com/pwollstadt/IDTxl.git

RUN python -c "from idtxl.multivariate_te import MultivariateTE"
RUN python -c "from idtxl.data import Data"

WORKDIR $HOME
RUN sudo apt-get update \
	&& sudo apt-get install -y --no-install-recommends \
		ed \
		less \
		locales \
		vim-tiny \
		wget \
		ca-certificates \
		fonts-texgyre \
	&& sudo rm -rf /var/lib/apt/lists/*

USER root
## Configure default locale, see https://github.com/rocker-org/rocker/issues/19
RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
	&& locale-gen en_US.utf8 \
	&& /usr/sbin/update-locale LANG=en_US.UTF-8
#USER jovyan
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8

## Use Debian unstable via pinning -- new style via APT::Default-Release
RUN echo "deb http://http.debian.net/debian sid main" > /etc/apt/sources.list.d/debian-unstable.list \
	&& echo 'APT::Default-Release "testing";' > /etc/apt/apt.conf.d/default

ENV R_BASE_VERSION 3.4.4

## Now install R and littler, and create a link for littler in /usr/local/bin
## Also set a default CRAN repo, and make sure littler knows about it too
WORKDIR /etc/R
#RUN rm Rprofile.site
WORKDIR $HOME
RUN apt-get update \
	&& apt-get install -t unstable -y --no-install-recommends \
		littler \
                r-cran-littler \
		r-base=${R_BASE_VERSION}-* \
		r-base-dev=${R_BASE_VERSION}-* \
		r-recommended=${R_BASE_VERSION}-*
#&&

RUN echo 'options(repos = c(CRAN = "https://cloud.r-project.org/"), download.file.method = "libcurl")' >> /etc/R/Rprofile.site \
        && echo 'source("/etc/R/Rprofile.site")' >> /etc/littler.r \
	&& ln -s /usr/share/doc/littler/examples/install.r /usr/local/bin/install.r \
	&& ln -s /usr/share/doc/littler/examples/install2.r /usr/local/bin/install2.r \
	&& ln -s /usr/share/doc/littler/examples/installGithub.r /usr/local/bin/installGithub.r \
	&& ln -s /usr/share/doc/littler/examples/testInstalled.r /usr/local/bin/testInstalled.r \
	&& install.r docopt \
	&& rm -rf /tmp/downloaded_packages/ /tmp/*.rds \
&& sudo rm -rf /var/lib/apt/lists/*
RUN sudo chown -R jovyan $HOME

RUN sudo /opt/conda/bin/pip install natsort
RUN sudo /opt/conda/bin/pip install pyspike pyNN lazyarray


WORKDIR /opt/conda/lib/python3.5/site-packages/pyNN/neuron/nmodl
RUN nrnivmodl

RUN sudo chown -R jovyan $HOME

RUN R -e 'install.packages(c("rPython"))'