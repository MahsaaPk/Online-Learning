# Online-Learning

## Introduction
This project explores two learning scenarios: Batch Learning and Online Learning. Batch Learning involves training a static Machine Learning model on a predefined dataset, where the entire training dataset is processed in one step. On the other hand, Online Learning, or incremental learning, continuously updates a dynamic model as new data points or small batches of data arrive.

## Selected Models
Four different models are used in this study.
Static models:
•
RandomForest (RF)
•
XGBoost (XGB)
•
Stochastic Gradient Descent (SGD)
•
PassiveAggressiveClassifier (PAC)
The optimal model identified during the batch learning phase, is chosen to be used for online learning task too.
## Methodology
### Data Preprocessing
Distribution and Skewness (statistical analysis of data): Examination of feature and target value distributions, along with skewness calculations, reveals skewness in some features. Figure 1 corroborates the skewness and is brought in the appendix. Log transformation proves ineffective for certain features, such as 'upper'. However, Box-Cox transformation yields better results, although some features like 'upper' and 'subdomain' remain skewed. Other transformation methods, including Square Root, Cube Root, Reciprocal, and Yeo-Johnson, were explored, but Box-Cox is identified as the most reliable. Importantly, the skewness of some features does not significantly impact model performance; F1 scores remained largely consistent even without transformations.
Dropping Irrelevant Features: The decision to drop the 'timestamp' feature is based on its negligible impact on prediction, as it represents time.
Missing Data(data cleansing): Eight missing data points in the 'longest_word' feature are deleted, considering the limited impact on the overall dataset.
Categorical Features: The dataset contains two mixed-type features with both numerical and categorical values. The hashing method is employed to convert each categorical value to a unique integer value of the same length. This approach is chosen over 'One Hot Encoding,' which would have generated too many features, and 'Label Encoding,' which is less efficient for nominal categories lacking meaningful order.
Data Splitting: Subsequently, the dataset is split into X_train, X_test, y_train, and y_test. The purpose is to assess the model's performance on data not encountered during training. Due to an adequate number of instances, I opted to allocate 25% of the data for the test set, following a standard convention.
Feature Selection: Four feature selection methods are employed and tested on the static models.
Feature Selection 1: The VarianceThreshold class from scikit-learn, designed to remove features with low variance, is utilized to eliminate features that do not contribute significantly to prediction.
Feature Selection 2: An L1-based feature selection technique utilizing a linear SVM, implemented with scikit-learn's LinearSVC, SelectFromModel, and the L1 penalty. This method is a form of embedded feature selection, where feature selection is integrated into the model training process.
Feature Selection 3: "Tree-based Feature Importance" or "Tree-based Feature Selection" involves fitting an Extra Trees Classifier to the data, extracting feature importance, and selecting features based on a specific threshold. This step utilizes "Extra Trees Classifier" combined with "SelectFromModel" in scikit-learn.
Feature Selection 4: Feature importance method provided by a Random Forest classifier, utilizing scikit-learn's SelectFromModel module with a RandomForestClassifier to select features based on their importance scores, considering a threshold of 0.05 as more reliable.
Imbalanced data: Evaluation of the dataset's imbalance ratio revealed an 82% balanced dataset. Figure2 shows the imbalance ratio and is brought in the appendix. The Synthetic Minority Over-sampling Technique (SMOTE) is applied to balance the train set. This approach aims to prevent information leakage, as changes to the test set's distribution are prohibited to maintain its integrity for feature prediction.
Feature Scaling: Feature scaling is performed using StandardScaler. To prevent information leakage, the train set is used to learn the mean and standard deviation for each feature in the train data. These parameters are then applied to both the train set and test set during the transformation phase, using the learned parameters from the fitting phase.
### Hyperparameter Tuning
A specific function (HyperP_tuning) is implemented for hyperparameter tuning based on suitable parameters for each model. The grid-search algorithm is employed for hyperparameter tuning across all models.
## Experimental Setup
Performance metrics are F1 score, precision, and recall. Due to uneven class distribution, it is more logical to use F1 score as Accuracy is misleading, and the use of precision, and recall is for better evaluation.
### Batch Learning
Following the implementation of the four models with four feature selection techniques, sixteen states are evaluated using F1 score, precision, and recall. The results are compared, and the best static model is selected for use in online learning which is Random Forest using feature selection #1. Figure 3 corroborates the performance of static models and is brought in the appendix.
### Online Learning
For incremental learning, the best model from the previous task is loaded. Predictions are initially made on the first window of stream data, comprising 1000 examples. For each window of coming data, the model is first tested on both static and dynamic models and then, based on a trial-and-error threshold of 0.86, if the obtained F1 score is lower, the dynamic model will be retrained. This process of prediction, evaluation, and retraining continues as necessary. The performance of both the static model and dynamic model on data streams is evaluated after every 1000 data points, considering F1 score, precision, and recall. The results of both batch learning and online learning are plotted. Figure 4, figure 5, and figure 6 show the comparison between static and dynamic models based on F1 score, Precision, and Recall respectively and are brought in the appendix.
## Results:
### Batch Learning Results:
The results are after preprocessing and hyper parameter tuning. Figure 7 shows the performance metrics’ scores of static models and is brought in the appendix.

*Static models performance*
| Model                | F1 score | Precision | Recall  |
|----------------------|----------|-----------|---------|
| RF-Feature selection1| 0.8610   | 0.7561    | 0.9997  |
| RF-Feature selection2| 0.8609   | 0.7560    | 0.9996  |
| RF-Feature selection3| 0.8606   | 0.7560    | 0.9988  |
| RF-Feature selection4| 0.8606   | 0.7559    | 0.9990  |
| XGB-Feature selection1| 0.8609  | 0.7560    | 0.9997  |
| XGB-Feature selection2| 0.8609  | 0.7560    | 0.9988  |
| XGB-Feature selection3| 0.8606  | 0.7550    | 0.9981  |
| XGB-Feature selection4| 0.8606  | 0.7560    | 0.9990  |
| SGD-Feature selection1| 0.8589  | 0.7554    | 0.9954  |
| SGD-Feature selection2| 0.8595  | 0.7556    | 0.9966  |
| SGD-Feature selection3| 0.8363  | 0.7539    | 0.9388  |
| SGD-Feature selection4| 0.8569  | 0.7552    | 0.9903  |
| PAC-Feature selection1| 0.84019 | 0.7538    | 0.9488  |
| PAC-Feature selection2| 0.8573  | 0.7552    | 0.9914  |
| PAC-Feature selection3| 0.8460  | 0.7499    | 0.9703  |
| PAC-Feature selection4| 0.8585  | 0.7546    | 0.9954  |

*RF: Random Forest, XGB: XGBoost, SGD: SGDClassifier, PAC: PassiveAggressiveClassifier*

Upon evaluating the static models, it is observed that the overall performance across models is similar. However, Random Forest with feature selection #1 method demonstrates a slightly better performance. Consequently, this model is selected for use in online learning.
### Online Learning Results
The initial comparison of scores between static and dynamic models at the first iteration reveals identical results:
•
F1 score: 0.66
•
Precision: 0.73
•
Recall: 0.61
However, through retraining the dynamic model, there is a notable improvement in its performance metrics' scores. This signifies that the dynamic model adapts more effectively to new data through retraining. Figure 8 displays the performance of dynamic model over time and is brought in the appendix. In contrast, the performance of static model exhibits fluctuations over time and does not show significant improvement. Figure 9 corroborates the performance of static model over time and is brought in the appendix.
Advantages and Limitation
Analyzing the static model's performance across different windows, it is observed that the F1 score fluctuates, while the online learning method consistently demonstrates improvement. This suggests the presence of concept drift, wherein the static model struggles with new data exhibiting a different distribution. The phenomenon is visualized in Figures 8 and 9, available in the appendix. On the other hand, as the Online model is retrained on new data, it adapts to new concepts, iteratively and can perform better.
Despite the random forest model not being inherently tailored for online learning, this study reveals relatively good performance. However, further investigations, particularly considering varying window sizes and computational costs, are warranted.

## Appendix
<img src="https://github.com/MahsaaPk/Online-Learning/assets/138306478/73ae8a6d-74ed-45b2-a68a-f25ae35c7151" alt="Figure 1" width="500" height="400" />

*Figure 1*

<img src="https://github.com/MahsaaPk/Online-Learning/assets/138306478/9a2acf70-bf18-40a8-95c3-907e036c6559" alt="Figure 1" width="500" height="400" />

*Figure 2*

<img src="https://github.com/MahsaaPk/Online-Learning/assets/138306478/19ca9a28-e884-436f-bc2c-d126e7146fc6" alt="Figure 1" width="500" height="400" />

*Figure 3*

<img src="https://github.com/MahsaaPk/Online-Learning/assets/138306478/6a41f4dc-1995-49b9-997d-39aec2d34500" alt="Figure 1" width="500" height="400" />

*Figure 4*

<img src="https://github.com/MahsaaPk/Online-Learning/assets/138306478/545470d8-1366-46e1-b5ba-323ff235cf24" alt="Figure 1" width="500" height="400" />

*Figure 5*

<img src="https://github.com/MahsaaPk/Online-Learning/assets/138306478/ae7e1d9b-32db-4fdd-aa44-758fc52fb525" alt="Figure 1" width="500" height="400" />

*Figure 6*

<img src="https://github.com/MahsaaPk/Online-Learning/assets/138306478/70644559-1b87-4d32-a3e5-fc9f38e2e8e5" alt="Figure 1" width="500" height="400" />

*Figure 7*

<img src="https://github.com/MahsaaPk/Online-Learning/assets/138306478/cd37ca11-8587-4806-9126-b13fb72f67bb" alt="Figure 1" width="500" height="400" />

*Figure 8*

<img src="https://github.com/MahsaaPk/Online-Learning/assets/138306478/af032842-6889-4b3a-8bfb-8d26f01697e1" alt="Figure 1" width="500" height="400" />

*Figure 9*







