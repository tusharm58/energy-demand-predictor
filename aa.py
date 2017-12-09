import os
import io,zipfile,tarfile
from pyspark import SparkContext
from pprint import pprint
import numpy as np
from numpy import genfromtxt
from pyspark.sql import SparkSession
import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, zscore
import matplotlib.pyplot as plt
def get_np_csv(zfBytes):
 #given a zipfile as bytes (i.e. from reading from a binary file),
# return a np array of rgbx values for each pixel
     name = zfBytes[0]
     zfBytes = zfBytes[1]
     d =dict()
     bytesio = io.BytesIO(zfBytes)
     tfiles = tarfile.open(fileobj=bytesio,mode="r")
     #find tif:
     pprint("control here")
     for fn in tfiles.getnames():

            if fn[-4:] == '.csv':#found it, turn into array:
                    #pprint(name + "/" + fn)
                    #tif = TiffFile(io.BytesIO(zfiles.open(fn).read()))
                    obj=tfiles.extractfile(fn)
                    #pprint(obj)
                    my_data = genfromtxt(obj, delimiter=',',skip_header=True)
                    #pprint(my_data.shape)
                    my_data = my_data[:8736, :3]
                    pprint(my_data.shape)
                    my_data = np.split(my_data, 52)
                    for i, x in enumerate(my_data):
                         my_data[i] = np.sum(my_data[i], axis=0)
                    my_data = np.array(my_data)
                    #pprint(fn)
                    fn1= "".join(fn.split("/")[1].split('_')[:-1])
                    fn1=fn1[:fn1.find("2004")]
                    fn2 = fn.split("/")[0][0:6]
                    fn=fn1+fn2
                    #pprint(fn)
                    d[fn]=my_data

     return list(d.items())
def map1(kv):
    l=[]
    pprint(kv)
    for x in kv[1]:
        l.append(x)
    return l
sc=SparkContext("local[*]","aa")
path="C:/Users/esais/Desktop/big_data/project/21/"
path1="C:/Users/esais/Desktop/big_data/project/2/"
#rdd1=sc.pickleFile(path1+"temp").union(sc.pickleFile(path1+"temp2")).union(sc.pickleFile(path1+"temp3")).union(sc.pickleFile(path1+"temp4")).union(sc.pickleFile(path1+"temp5")).union(sc.pickleFile(path1+"temp6")).union(sc.pickleFile(path1+"temp7")).union(sc.pickleFile(path1+"temp8")).union(sc.pickleFile(path1+"temp9")).union(sc.pickleFile(path1+"temp10"))
#final_rdd=rdd1.reduceByKey(lambda x,y : np.add(x,y))
def evaluateBetasOverTest(betas, X_test, y_test):
 y_pred = np.matmul(X_test, betas)[:,0]
 print(len(y_pred))
 plt.plot(range(41,51),y_pred.tolist())
 plt.show()
 print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
 print("Pearson Correlation:", pearsonr(y_test, y_pred))

rdd1=sc.pickleFile(path1+"part10")
kv = rdd1.collect()
kv1=[]
for i,x in enumerate(kv):

    kv1.append([kv[i][0],kv[i][1].tolist()])
#pprint(kv1)
data = kv1[0][1]

data = np.asarray(data)
data=data[0:,1]
price=data[3:]
plt.plot(range(52),data.tolist())
xt1=data[2:-1]
xt2=data[1:-2]
xt3=data[0:-3]
print(xt1.shape,xt2.shape,xt3.shape)
features = np.vstack((xt1,xt2,xt3)).transpose()
#print(features)
featuresZ = zscore(features)
#pprint(featuresZ[:5])
featuresZ_pBias = np.c_[np.ones((featuresZ.shape[0], 1)), featuresZ]
offset = int(featuresZ_pBias.shape[0] * 0.8)
featuresZ_pBias_test, price_test = featuresZ_pBias[offset:], price[offset:]
featuresZ_pBias, price = featuresZ_pBias[:offset], price[:offset]
#X = tf.constant(featuresZ_pBias, dtype=tf.float32, name="X")
#y = tf.constant(price.reshape(-1,1), dtype=tf.float32, name="y")

#Xt = tf.transpose(X)
#penalty = tf.constant(1.0, dtype=tf.float32, name="penalty")
#I = tf.constant(np.identity(featuresZ_pBias.shape[1]), dtype=tf.float32, name="I")

#beta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(Xt, X) + penalty*I), Xt), y)

### everything above is just definitions of operations to carry out over tensors
### now let's create a session to run the operations
#with tf.Session() as sess:
#  beta_value = beta.eval()

#print(beta_value)
#evaluateBetasOverTest(beta_value, featuresZ_pBias_test, price_test)
finaly = []
def testL2Reg(penalty_value = 1000, learning_rate = 0.001, n_epochs = 300):
 X = tf.constant(featuresZ_pBias, dtype=tf.float32, name="X")
 y = tf.constant(price.reshape(-1,1), dtype=tf.float32, name="y")
 Xt = tf.transpose(X)
 penalty = tf.constant(penalty_value, dtype=tf.float32, name="penalty")
# I = tf.constant(np.identity(featuresZ_pBias.shape[1]), dtype=tf.float32, name="I")
 beta = tf.Variable(tf.random_uniform([featuresZ_pBias.shape[1], 1], -1., 1.), name = "beta")
 y_pred = tf.matmul(X, beta, name="predictions")
 penalizedCost = tf.reduce_sum(tf.square(y - y_pred)) #+ penalty * tf.reduce_sum(tf.square(beta))
 optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
 training_op = optimizer.minimize(penalizedCost)
 init = tf.global_variables_initializer()
 with tf.Session() as sess:
  sess.run(init)
  for epoch in range(n_epochs):
   if epoch %10 == 0: #print debugging output
    print("Epoch", epoch, "; penalizedCost =", penalizedCost.eval())
   sess.run(training_op)
  #done training, get final beta:
  best_beta = beta.eval()
  pred = y_pred.eval()
  plt.plot(range(3,42),pred)

 print(best_beta)
 evaluateBetasOverTest(best_beta, featuresZ_pBias_test, price_test)
testL2Reg(1)
#pprint(featuresZ_pBias[:5])
#pprint(data)


#values=np.asarray(values)
#np.savetxt(path1+"foo.csv", values,fmt='%.64f', delimiter=",")
#spark = SparkSession(sc).getOrCreate()
#df = spark.createDataframe(rdd1)

#df.write.csv(path1+"file.csv", sep=',', header=False)
#rdd1.collect()
#zip_rdd = sc.binaryFiles(path+"*")
#np_rdd=zip_rdd.flatMap(get_np_csv)
##countywise weekly data
#np_rdd = np_rdd.map(lambda x : ["".join(x[0].split("/")[1].split('_')[:-1]),x[1]])
#state_rdd = np_rdd.reduceByKey(lambda x,y : np.add(x,y))
#state_rdd.saveAsPickleFile(path1+"part2")
#with open("temp.txt","w") as f:
    #print(state_rdd.collect(),file=f)