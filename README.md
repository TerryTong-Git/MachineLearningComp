# MachineLearningComp
In this repository, I detail the steps I took to reproduce the results I obtained in my Kaggle Submission. Most of it was inspired by the Catboost tutorials: [[github](https://github.com/catboost/tutorials/blob/master/python_tutorial.ipynb)]

## The main gist
The main strategy employed was a gradient boosted decision tree with hyperparameters selected by Bayesian Optimization. The final model was an ensemble of the best performing models selected with cross-validation on the ROC AUC with the false positive rate at 0.01. I trained 300 models, and ensembled the top 30. 

### Exploratory Data Analysis
On the explaratory data analysis, I saw that the influences of each parameter was not negligible, so did not delete them. But in hindsight, this may have not been the best choice. 
![alt text](https://github.com/TerryTong-Git/MachineLearningComp/blob/main/PREDA_results.png?raw=true)

### Parameter Tuning
For the hyperparameters, I searched over the learning rate, l2 leaf regularization, model size regularization, and the depth of the tree. The hyperparameters were selected on a 5 fold cv, with each fold holding 75% of the train set data. I set the loss here to cross-entropy rather than log-loss, but they are essentially the same. For the evaluation metric, I use the precision recall area under curve of the reciever operating curve, hence forth referred to as the PRAUC. 

### Final Ensemble Model Selection
Here I wanted to maximize the amount of data used for training the model, so I split the train test set to 90% to 10% respectively. These splits were done over multiple seeds, each resulting in a model that is evaluated on the test set. Then, I select the best 30 models and ensembled them by simply averaging their predictions. This yeilded a 0.2% improvement over just choosing the top model on the hidden test set. 

### Results
On the local test set I achieve 97% PRAUC, since I am trying to maximize the precision and recall. On Kaggle, this translates to 93.96% on the hidden test set. 

Here is a picture of the metrics. 
![alt text](https://github.com/TerryTong-Git/MachineLearningComp/blob/main/PRAUC.png?raw=true)

### Improvements 
In hindsight, I should have done more feature engineering. on the EDA, I saw that each variable had some influences, so I did not think to delete them. I should have deleted them. If I had more time, I would ensemble the models by not just averaging their predictions, but taking their predictions and training a logistic regression model over them, following the idea from the winning kaggle solution from : [[santander comp](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003)]


### What did not work
What did not work: 
* neural network with 8 layers, and 30 input features, hidden dimensions 1000, and the final layer is a logistic function to output the probabilities. 
* support vector machine with linear, rbf and polynomial kernel. The reason behind this is that we are attempting to calibrate the probabilities well, and SVM is poor compared to methods with a logistic function. 
* XGBoost gradient boosting works mediocrely. This was the most annoying to train because the hyperparameter search took very long even over the GPUs. 
* Best performing was the CatBoost gradient boosting. Despite not using categorical features, this framework made training, EDA and feature engineering easy. I tried ensembling over 10,20,30 models. The more the better from the results, but the returns are marginal. 


