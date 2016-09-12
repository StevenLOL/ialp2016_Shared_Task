import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold,train_test_split
from sklearn.svm import LinearSVR,SVR
from sklearn.linear_model import SGDRegressor,LinearRegression,HuberRegressor,RANSACRegressor,TheilSenRegressor,RandomizedLogisticRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import RobustScaler,MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,BaggingRegressor,AdaBoostRegressor
from sklearn.neighbors import RadiusNeighborsRegressor,KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.decomposition import MiniBatchSparsePCA,PCA, TruncatedSVD,NMF
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectPercentile,chi2,mutual_info_classif,f_classif,f_regression
from sklearn.pipeline import Pipeline,FeatureUnion
#from sklearn.model_selection import
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec