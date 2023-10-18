
from sklearn.metrics import classification_report

import K_nearest_neighbours 
import logistic_regression
import naive_bayes

l1 = [K_nearest_neighbours.KNNmethod(),logistic_regression.LRmethod(),naive_bayes.NBmethod()]
 
l2 = [K_nearest_neighbours.report(),logistic_regression.report(),naive_bayes.report()]


for i in range(0,3):
    
    l1[i]
    l2[i]
