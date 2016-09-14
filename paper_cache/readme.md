Here is data folder for shared task.

#Files in the folder
\*.pickle are data files contrain training vectors, training target , test vectors, test ids

w.list is a list of training words


#Naming convention

**CWE_L_100_CB**

**CWE_L** denotes feature type

**100** denotes size of word vector

**CB** denotes it is trained with continuous bag-of-words model (CBOW)
 
**SG** denotes Skip-Gram





#Usage

```
#this will give you the training vectors: trainx, training target: trainy , test vectors: testx , test ids: testids


trainx,trainy,testx,testids=pickle.load(open('CWE_P_300_SG.picke','rb'))

#here all the trainy are valence values, replace them with arousal target values (refer to the w.list word list ) if you want estimate the arousal value.



```

