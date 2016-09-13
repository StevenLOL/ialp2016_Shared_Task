import pandas as pd
import numpy as np
import os
import warnings
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
from sklearn.preprocessing import StandardScaler
import cPickle as pickle


fiels_wordvectors=[

    #    ('loadFasttextWord2Vector','/data1/ML_DATA/NLP/w2v/IALP/w2v_100_CB.txt'),
    #('loadFasttextWord2Vector','/data1/ML_DATA/NLP/w2v/IALP/w2v_100_SG.txt'),
    # ('loadFasttextWord2Vector','/data1/ML_DATA/NLP/w2v/IALP/w2v_300_CB.txt'),
 ('loadFasttextWord2Vector','/data1/ML_DATA/NLP/w2v/IALP/w2v_300_SG.txt'),



    #('loadFasttextWord2Vector','/data1/ML_DATA/NLP/w2v/IALP/CWE_P_100_CB.txt'),
    #('loadFasttextWord2Vector','/data1/ML_DATA/NLP/w2v/IALP/CWE_P_100_SG.txt'),
    # ('loadFasttextWord2Vector','/data1/ML_DATA/NLP/w2v/IALP/CWE_P_300_CB.txt'),
    ('loadFasttextWord2Vector','/data1/ML_DATA/NLP/w2v/IALP/CWE_P_300_SG.txt'),



    #('loadFasttextWord2Vector','/data1/ML_DATA/NLP/w2v/IALP/CWE_L_100_CB.txt'),
    # ('loadFasttextWord2Vector','/data1/ML_DATA/NLP/w2v/IALP/CWE_L_100_SG.txt'),
    #('loadFasttextWord2Vector','/data1/ML_DATA/NLP/w2v/IALP/CWE_L_300_CB.txt'),
    ('loadFasttextWord2Vector','/data1/ML_DATA/NLP/w2v/IALP/CWE_L_300_SG.txt'),

    #('loadFasttextWord2Vector','/data1/ML_DATA/NLP/w2v/IALP/fasttext_100_CB.vec'),
    #('loadFasttextWord2Vector','/data1/ML_DATA/NLP/w2v/IALP/fasttext_100_SG.vec'),
    #('loadFasttextWord2Vector','/data1/ML_DATA/NLP/w2v/IALP/fasttext_300_CB.vec'),
    ('loadFasttextWord2Vector','/data1/ML_DATA/NLP/w2v/IALP/fasttext_300_SG.vec'),


]


def evalTrainData(trainDatax,trainV,random_state=2016,eid=''):
    fold=10

    scores=[]
    pccs=[]
    cvcount=0
    assert len(trainDatax)==len(trainV)
    import time
    for roundindex in range(0,3):
        #random_state=random_state+roundindex
        skf=KFold(fold,shuffle=True,random_state=random_state+roundindex)
        for trainIndex,evalIndex in skf.split(trainDatax):
            t1=time.time()
            cvTrainx,cvTrainy=trainDatax[trainIndex],trainV[trainIndex]
            cvEvalx,cvEvaly=trainDatax[evalIndex],trainV[evalIndex]
            #print cvTrainx.shape,cvEvalx.shape
            #scaler=StandardScaler()
            #cvTrainy=scaler.fit_transform(cvTrainy)
            ###pca=PCA(n_components=100)
            ###cvTrainx=pca.fit_transform(cvTrainx)
            ###cvEvalx=pca.transform(cvEvalx)
            ##cvTrainy=np.log(cvTrainy+1)
            ###lsvr=AdaBoostRegressor(base_estimator=MLPRegressor(random_state=random_state,early_stopping=True,max_iter=2000)
            ###                                                 ,n_estimators=30,learning_rate=0.01)

            lsvr=getxlf(random_state=random_state)
            ####XGBOOOST###
            ###xgtrainx,xgtestx,xgtrainy,xgtesty=train_test_split(cvTrainx,cvTrainy,train_size=0.9,random_state=random_state)
            ###lsvr.fit(xgtrainx,xgtrainy,eval_set=[(xgtestx,xgtesty)],  eval_metric='mae',verbose=False,early_stopping_rounds=1)
            ####XGBOOOST END###
            lsvr.fit(cvTrainx,cvTrainy)
            predict=lsvr.predict(cvEvalx)
            ##predict=np.exp(predict)-1
            #predict=scaler.inverse_transform(predict)
            score=mean_absolute_error(cvEvaly,predict)
            score2=mean_squared_error(cvEvaly,predict)
            pcc=np.corrcoef(cvEvaly,predict)[0, 1]
            print (cvcount,'MAE',score,'PCC',pcc,time.time()-t1,time.asctime( time.localtime(time.time()) ) ,'Train sahpe:',cvTrainx.shape,'eval sahpe:', cvEvalx.shape)
            scores.append(score)
            pccs.append(pcc)
            cvcount+=1

    print ('###',eid,'MAE',np.mean(scores),'PCC',np.mean(pccs))


pca_n_components=100
def getxlf(random_state=2016):
    xlf1= Pipeline([
                          ('svd',PCA(n_components=pca_n_components)),
                          ('regressor',AdaBoostRegressor(

                        base_estimator=  MLPRegressor(random_state=random_state,early_stopping=True,max_iter=2000)
                                                         ,n_estimators=30,learning_rate=0.01)),
                          ])
    xlf2=Pipeline([
                          #('svd',PCA(n_components=pca_n_components)),
                          #('norm',StandardScaler()),
                          ('regressor', MLPRegressor(random_state=random_state,early_stopping=True,max_iter=2000)
                           #GradientBoostingRegressor()
                           #MLPRegressor(random_state=random_state,early_stopping=True,max_iter=2000)
                           #XGBRegressor()
                          )
                          ])
    xlf4=Pipeline([
                          #('svd',PCA(n_components=pca_n_components)),
                          #('norm',StandardScaler()),
                          ('regressor', GradientBoostingRegressor(random_state=random_state)
                           #LinearSVR()
                           #GradientBoostingRegressor()
                           #MLPRegressor(random_state=random_state,early_stopping=True,max_iter=2000)
                           #XGBRegressor()
                          )
                          ])

    xlf3=XGBRegressor(max_depth=10,
                        learning_rate=0.1,
                        n_estimators=1000,
                        silent=True,
                        objective='reg:linear',
                        nthread=4,
                        gamma=0.001,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=random_state,
                        missing=None)

    return xlf4


def processTraining(emid=0,targetfilename='./paper_cache/pv01.pickle'):
    fid=fiels_wordvectors[emid][1].split('/')[-1].split('.')[0]
    print (fid)
    tempfile='./paper_cache/'+fid+'.picke'

    if os.path.isfile(tempfile):
        print ('load from cache')
        ftrainx,ftrainv,testWordInChs,testids=pickle.load(open(tempfile,'rb'))
        evalTrainData(ftrainx,ftrainv,random_state=2016,eid=fid)


for eid in range(0,len(fiels_wordvectors)):
    processTraining(emid=eid)