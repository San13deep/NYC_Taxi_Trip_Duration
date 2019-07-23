#!/bin/sh
# install xgboost manually 
# https://stackoverflow.com/questions/43327020/xgboostlibrarynotfound-cannot-find-xgboost-library-in-the-candidate-path-did-y
# http://xgboost.readthedocs.io/en/latest/build.html
# will need to install gcc@7 (or gcc@5) before install xgboost
cd ~ 
git clone --recursive https://github.com/dmlc/xgboost.git
cd xgboost
./build.sh
cd python-package
python setup.py install