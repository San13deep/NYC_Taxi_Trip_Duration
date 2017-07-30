### Utility scripts for trip duration prediction 


## Tech
- python 3, sklearn

## Scripts 

```
prepare.py :  preparing data (feature extract / data cleaning...) 
train.py   :  training model 
model.py   :  model IO 

```

## Quick start 

```
$export PYTHONPATH=/Users/yennanliu/NYC_Taxi_Trip_Duration/
# add script route to PYTHONPATH
$python run/run_train.py
# fitting model with train data, save model as pickle file 
# model/model_0726.pkl
$python run/run_test.py
# predict test data with saved model, save prediction csv
# output/submit0726.csv

```





