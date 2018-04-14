#!/bin/bash
git clone https://github.com/jlizier/jidt
cd jidt
sudo ant build
sudo apt-get install python-jpype
sudo /opt/conda/bin/pip install JPype1 pyspike natsort
sudo /opt/conda/bin/pip install git+https://github.com/pwollstadt/IDTxl.git
python -c "from idtxl.multivariate_te import MultivariateTE"
python -c "from idtxl.data import Data"
