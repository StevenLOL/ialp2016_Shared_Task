# ialp2016_Shared_Task

Here is the systems for following paper, downoad the code and data , one could reproduce all the results.

Aicyber’s System for [IALP 2016 Shared Task](http://nlp.innobic.yzu.edu.tw/tasks/
dsa_w/): Character-enhanced Word Vectors and Boosted Neural Networks

#Prerequisites

##Scikit-learn
Install the latest [scikit-learn](https://github.com/scikit-learn/scikit-learn). This will provide most of the regressors and machine learning utilities.

##xgboost
Install the xgboost via:
```
sudo pip install xgboost
```


#Evaluations

##Features VS LSVR baseline
```
python table_1_LSVR.py
```
Sample output, after 3 rounds of 10 folds cross-validation, the 'MAE', 0.936 is the first value in the Table 1:
```
w2v_100_CB
load from cache
(0, 'MAE', 0.95422037820373351, 'PCC', 0.72929625905287654, 0.24031305313110352, 'Wed Sep 14 11:41:14 2016', 'Train sahpe:', (1486, 100), 'eval sahpe:', (166, 100))
(1, 'MAE', 0.88660967155777792, 'PCC', 0.79691459045767477, 0.22659897804260254, 'Wed Sep 14 11:41:14 2016', 'Train sahpe:', (1486, 100), 'eval sahpe:', (166, 100))
(2, 'MAE', 0.92380137358766068, 'PCC', 0.73167313941310941, 0.23987698554992676, 'Wed Sep 14 11:41:15 2016', 'Train sahpe:', (1487, 100), 'eval sahpe:', (165, 100))
(3, 'MAE', 0.97292950366904318, 'PCC', 0.72487931346684542, 0.24808311462402344, 'Wed Sep 14 11:41:15 2016', 'Train sahpe:', (1487, 100), 'eval sahpe:', (165, 100))
(4, 'MAE', 0.91984530597533742, 'PCC', 0.78713717096390567, 0.2524690628051758, 'Wed Sep 14 11:41:15 2016', 'Train sahpe:', (1487, 100), 'eval sahpe:', (165, 100))
...
...
('###', 'w2v_100_CB', 'MAE', 0.93628075980941383, 'PCC', 0.75431909093171989)
...
...
```
##Features VS BNN
```
python ./table_2_BNN.py
```
##LSVR, BNN VS NN GBM XGB
```
python ./table_3_NN.py
python ./table_3_GBM.py
python ./table_3_XGB.py
```

## PCA100 BNN VS BNN_Norm

```
python ./table_4_PCA100_BNN.py
python ./table_4_PCA100_BNN_Norm.py
```



