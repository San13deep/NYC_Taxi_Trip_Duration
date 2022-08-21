<p align="center"><img src ="https://github.com/yennanliu/NYC_Taxi_Trip_Duration/blob/master/data/nyc_taxi.jpg"></p>

<h1 align="center"><a href="https://www.kaggle.com/c/nyc-taxi-trip-duration">NYC Taxi Trip Duration</a></h1>

<p align="center">
<!--- travis -->
<a href="https://travis-ci.org/yennanliu/NYC_Taxi_Trip_Duration"><img src="https://travis-ci.org/yennanliu/NYC_Taxi_Trip_Duration.svg?branch=master"></a>
<!--- coverage status -->
<a href='https://coveralls.io/github/yennanliu/NYC_Taxi_Trip_Duration?branch=master'><img src='https://coveralls.io/repos/github/yennanliu/NYC_Taxi_Trip_Duration/badge.svg?branch=master' alt='Coverage Status' /></a>
<!--- PR -->
<a href="https://github.com/yennanliu/NYC_Taxi_Trip_Duration/pulls"><img src="https://img.shields.io/badge/PRs-welcome-6574cd.svg"></a>
<!--- notebooks mybinder -->
<a href="https://mybinder.org/v2/gh/yennanliu/NYC_Taxi_Trip_Duration/master"><img src="https://img.shields.io/badge/launch-Jupyter-5eba00.svg"></a>
</p>

## INTRO

>Predict the total ride duration of taxi trips in New York City. primary dataset is one released by the NYC Taxi and Limousine Commission, which includes pickup time, geo-coordinates, number of passengers, and several other variables.

Please download the train data via : https://www.kaggle.com/c/nyc-taxi-trip-duration/data, and save at `data/train.csv`. Then 
you should be able to run the ML demo code (scripts under `run/`)

* [Kaggle Page](https://www.kaggle.com/c/nyc-taxi-trip-duration)
* [Analysis nb](https://nbviewer.jupyter.org/github/yennanliu/NYC_Taxi_Trip_Duration/blob/master/notebook/NYC_Taxi_EDA_V1_Yen.ipynb) - EDA ipython notebook 
* [ML nb](https://nbviewer.jupyter.org/github/yennanliu/NYC_Taxi_Trip_Duration/blob/master/notebook/NYC_Taxi_ML_V1_Yen.ipynb) - ML ipython notebook 
* [Main code](https://github.com/yennanliu/NYC_Taxi_Trip_Duration/tree/master/run) - Final ML code in python 
* [Team member repo](https://github.com/juifa-tsai/NYC_Taxi_Trip_Duration)

> Please also check [NYC_Taxi_Pipeline](https://github.com/yennanliu/NYC_Taxi_Pipeline) in case you are interested in the data engineering projects with similar taxi dataset. 

## FILE STRUCTURE

```
├── README.md
├── data     : directory of train/test data 
├── documents: main reference 
├── model    : save the tuned model
├── notebook : main analysis
├── output   : prediction outcome
├── reference: other reference 
├── run      : fire the train/predict process 
├── script   : utility script for data preprocess / feature extract / train /predict  
├── spark_   : Re-run the modeling with SPARK Mlib framework : JAVA / PYTHON / SCALA
└── start.sh : launch training env
```

## QUICK START

## PROCESS

```
EDA -> Data Preprocess -> Model select -> Feature engineering -> Model tune -> Prediction ensemble
```

```
# PROJECT WORKFLOW 

#### 1. DATA EXPLORATION (EDA)

Analysis : /notebook  

#### 2. FEATURE EXTRACTION 

2-1. **Feature dependency**
2-2. **Encode & Standardization** 
2-3. **Feature transformation** 
2-4. **Dimension reduction** ( via PCA) 

Script : /script 
Modeling : /run 

#### 3. PRE-TEST

3-1. **Input all standardized features to all models** <br>
3-2. **Regression**

#### 4. OPTIMIZATION

4-1. **Feature optimization** 
4-2. **Super-parameters tuning** 
4-3. **Aggregation**

#### 5. RESULTS 

-> check the output csv, log  
```
---

## Development 
```bash 
# unit test 
$ export PYTHONPATH=/Users/$USER/NYC_Taxi_Trip_Duration/
$ pytest -v tests/
# ============================== test session starts ==============================
# platform darwin -- Python 3.6.4, pytest-5.0.1, py-1.5.2, pluggy-0.12.0 -- /Users/jerryliu/anaconda3/envs/yen_dev/bin/python
# cachedir: .pytest_cache
# rootdir: /Users/jerryliu/NYC_Taxi_Trip_Duration
# plugins: cov-2.7.1
# collected 5 items                                                               

# tests/test_data_exist.py::test_training_data_exist PASSED                 [ 20%]
# tests/test_data_exist.py::test_validate_data_exist PASSED                 [ 40%]
# tests/test_udf.py::test_get_haversine_distance PASSED                     [ 60%]
# tests/test_udf.py::test_get_manhattan_distance PASSED                     [ 80%]
# tests/test_udf.py::test_get_direction PASSED                              [100%]

# =========================== 5 passed in 2.68 seconds ===========================
```
## REFERENCE

- XGBoost
  - http://xgboost.readthedocs.io/en/latest/python/python_api.html 
  - https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
  - Core XGBoost Library VS scikit-learn API
  	- https://zz001.wordpress.com/2017/05/13/basics-of-xgboost-in-python/

- LightGBM
  - https://github.com/Microsoft/LightGBM/wiki/Installation-Guide
  Summary:
The goal of this project is to predict trip duartion. Its important to get same or near match score for Training and Testing input. Otherwise, getting good score on Training set and poor on Testing set is not good for predictions in real world. In this project using several Regression models, Taxi Trip duration predicted for NYC. Four ML algorithms used for this project. Linear Regression, Decision Tree, Random Forest and XG Boost. Linear Regression performed very poorly comparatively to the other models. Decision Tree gave overfitted model so using Hyperparameters tuning models performance and accuracy stabilised. The models performance evaluated using MSE, RMSE and R2 score metrics. Its reccomended to have lower score for MSE and RMSE. Ideally 1 - R2 score means model capturing underlaying pattern efficeintly and 100% accurately.

Conclusion's:
Linear Regression accuracy performance was poor. 0.60 R2 score for both Traina and Test.
Decision Tree, Random Forest and XG Boost gave better accuracy than Linear Regression.
XG Boost predicted good accurcy R2 score, 0.77 on Train and 0.76 on Test set.
Future Scope This dataset has many inconsistent and inaccurate records. The accuracy and performance can be further increased by removing more inconsistent data as it contains more than 14Lac records.

# NYC Taxi Trip Prediction
# Problem Description
Task is to build a model that predicts the total ride duration of taxi trips in New York City. Primary dataset is one released by the NYC Taxi and Limousine Commission, which includes pickup time, geo-coordinates, number of passengers, and several other variables.

<h2> :floppy_disk: Project Files Description</h2>

<p>This repository includes 1 colab notebook (.ipynb) file and 1 pdf file of project presentation. </p>
<h4>About Files:</h4>
<ul>
<li><b>NYC Taxi Trip Time Prediction Capstone Project.ipynb</b> - This file includes Features description, exploratory data Analysis, feature engineering, feature scaling and implemented algorithms for eg. <b>Linear Regression, Decision Tree, XGBoost.</b></li> 
 <li><b>NYC_PPT</b> -  This is a power point presentation file of a project. It includes various visualaized plots of EDA using <b>Seaborn and Matplotlib</b>. The result chart of various implemented algorithms.</li>
  

![---------------------------------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :clipboard: Summary</h2>
<p align="justify">The given dataset conatins more than 14 lac records and 11 columns. The main goal of this project to predict the trip duration. To get more insights about the dataset, performed Exploratory data analysis to understand the main characteristics of various features. Using existing features, created new features for model building and to interpret data more easily. After analyzing the dataset, it’s found that there is a some inconsistency in some recorded data. The passenger count was high like 7 to 9 passengers in taxi. The travelled distance is very less but trip duration is quite high. There were many outliers in many columns and after analyzing the data, those outliers removed and dataset prepared for model building. Various algorithms apllied on prepared dataset afetr train and test split of dataset. <b>Linear Regression algorithm performed very poorly on given dataset. It predicted 0.60 R2 score on train and test dataset. Decision Tree predicted 0.99 for Train dataset and 0.49 for test. It’s observed that its a overfitting model. XG Boost predicted good accuracy on train and test dataset. It predicted 0.77 and 0.76 R2 score.<b/></p>

![---------------------------------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- CREDITS -->
<h2 id="credits"> :scroll: Credits</h2>

<b>Saurabh Funde</b> | Avid Reader | Data Science enthusiast |

[![LinkedIn Badge](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/saurabhfunde/)
[![Medium Badge](https://img.shields.io/badge/Medium-1DA1F2?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@saurabh.f)


![---------------------------------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
<h2>Algorithm References:</h2>
<ul>
  <li><p>Linear Regression</p>
      <p>Link: https://towardsdatascience.com/linear-regression-using-python-b136c91bf0a2</p>
  </li>
  <li><p>Decision Tree</p>
      <p>Link: https://towardsdatascience.com/the-complete-guide-to-decision-trees-28a4e3c7be14</p>
  </li>
  <li><p>XG Boost</p>
      <p>Link: https://towardsdatascience.com/a-journey-through-xgboost-milestone-3-a5569c72d72b</p>
  </li>
  <li><p>Random Forest</p>
      <p>Link: https://www.analyticsvidhya.com/blog/2021/10/an-introduction-to-random-forest-algorithm-for-beginners/</p>
  </li>
 <li><p>Hyperparameter Tuning</p>
      <p>Link: https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680</p>
  </li>
</ul>
 
 <h2>Data Source</h2>
  <li><p>Kaggle</p>
      <p>Link: https://www.kaggle.com/c/nyc-taxi-trip-duration</p>

![---------------------------------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
