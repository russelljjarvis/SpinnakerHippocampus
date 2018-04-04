FROM russelljarvis/neuronunit
RUN echo "deb http://cran.rstudio.com/bin/linux/ubuntu xenial/" | sudo tee -a /etc/apt/sources.list
RUN gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
RUN gpg -a --export E084DAB9 | sudo apt-key add -
RUN apt-get update
RUN apt-get install r-base r-base-dev r-base-core
RUN update && sudo apt-get upgrade

USER jovyan
RUN sudo /opt/conda/bin/pip install psutil
ENV QT_QPA_PLATFORM offscreen
RUN sudo rm -rf /opt/conda/lib/python3.5/site-packages/neuronunit-0.1.8.8-py3.5.egg/neuronunit
RUN sudo rm -rf $HOME/neuronunit
COPY . $HOME/neuronunit

RUN pip install dask
RUN pip install distributed
WORKDIR $HOME

