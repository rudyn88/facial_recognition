# facial_recognition
An assumption in standard classification theory is that the input distributions of the test and the training
sets are identical. In reality, it is common for the machine to have unseen data (’out-of-distribution’ data)
that is different from the training data distribution (’in-distribution’ data). Therefore, building a trust-
worthy machine learning system is important specially in social justice problems, where data distribution
from under-represented group could be different from what is used in training a machine learning model
which may result in overconfidently wrong decisions or predictions.
The project focuses on developing a neural network that can differentiate data from different distributions.
We are interested in the following question: When there is no information to determine what is out-of-
distribution when training a model, how can we design a system that can detect anomalous inputs? We
will approach this problem by using by using Outlier Exposure Model where an auxiliary data set that is
different from either in- or out-of-distribution data is introduced to improve the generalization ability of the
neural networks. We further investigate how we can improve the model sensitivity with respect to OOD
data through Geometric Sensitivity Decomposition. We apply the methods on the facial information data
and improve the reliability of the machine learning algorithms across different racial and ethnic groups.
