Supplemental Digital Content 8 - R Code: Training and Testing ML Models

library(dplyr)
library(lattice)
library(ggplot2)
library(caret)
library(tidyr)
library(h2o) #scalable open source machine learning platform

#load your dataset
setwd(YOUR_DIRECTORY)
your_data<-read.csv("atOR.csv")
your_data<-your_data[ ,2:ncol(your_data)] #removing the additional index from the CSV, if needed

#confirm elimination of NA and remove cases greater than 1440 minutes
your_data<-your_data %>%
  filter(!is.na(Patient.In.Room.Duration)) %>%
  filter(Patient.In.Room.Duration>0) %>%
  filter(Patient.In.Room.Duration<1440)

#convert time of day to categorical
your_data$timeofday<-as.factor(your_data$timeofday)

#initialize H2O server
h2o.init(nthreads = -1)

#convert data to H2O format
your_data.h2o<-as.h2o(your_data)

##################
######Train Models
##################
#break data into testing and training sets
trainIndex = sample(1:nrow(your_data), size = round(0.7*nrow(your_data)),replace=FALSE)
train = your_data[trainIndex ,]
test = your_data[-trainIndex ,] 
train.h2o<-as.h2o(train)
test.h2o<-as.h2o(test)
y.dep<-"Patient.In.Room.Duration"
x.indep<-setdiff(names(train.h2o), y.dep)

#using your data, build a gbm model to predict in room duration
proclength_gbm <- h2o.gbm(y= y.dep, x = x.indep, training_frame = train.h2o,ntrees=500,nfolds=5,max_depth=5,learn_rate=0.1,stopping_tolerance=0.01,stopping_metric="MAE")
h2o.performance(proclength_gbm)
###saving proclength model
h2o.saveModel(proclength_gbm, path =YOUR_PATH, force = TRUE) #force is to overwrite any existing model; note the new folder I created
#shap summary plot for the model
shap_plot <- h2o.shap_summary_plot(proclength_gbm, train.h2o)
plot(shap_plot)
#local explanation of any predicted output, SHAP explanation shows contribution of features for a given instance.
#shapr works for H2O tree-based models, such as Random Forest, GBM and XGboost only.
shapr_plot <- h2o.shap_explain_row_plot(proclength_gbm, train.h2o, row_index = 1)
plot(shapr_plot)

##################
######Test Above Models
##################
#You can now make predictions of your test data or new data.  Using the proclength model as an example:
pred <- h2o.predict(object = proclength_gbm, newdata = test.h2o) #here I use the existing data which was used in training
pred
#you will need to tune your model's hyperparameters using, potentially using the h2o.grid function over hyperparameter ranges

##################
######Train New Models For Quantile Error Model
##################
#using your data and the previous model, build separate gbm models to predict the "Patient.In.Room.Duration" variable
y.dep<-"Patient.In.Room.Duration"
x.indep<-setdiff(names(train.h2o), y.dep)
hyper_params <- list(quantile_alpha = c(.2, .5, .8), ntrees=500,max_depth=5,learn_rate=0.1,stopping_tolerance=0.001,stopping_metric="MAE")
#grid searching can be used for tuning hyperparameters, in this instance we are using it to test quantiles of data
#https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/quantile_alpha.html
gridOR <- h2o.grid(x = x.indep, y = y.dep, training_frame = your_data.h2o, algorithm = "gbm",
                   grid_id = "gridname",
                   distribution = "quantile",
                   hyper_params = hyper_params,
                   nfolds=5)
sortedGrid <- h2o.getGrid("gridname", sort_by = "mae", decreasing = FALSE)
sortedGrid #show models
q5OR<-h2o.getModel(sortedGrid@model_ids[[1]]) #store models separately
q2OR<-h2o.getModel(sortedGrid@model_ids[[2]])
q8OR<-h2o.getModel(sortedGrid@model_ids[[3]])

#prediction interval based on quantile model
h2o.predict(q5OR, your_data)
h2o.predict(q2OR, your_data)
h2o.predict(q8OR, your_data)

#quantile prediction interval output data as a table
quantile_prediction_table->(cbind(h2o.predict(q5OR,new_data),h2o.predict(q2OR,new_data),h2o.predict(q8OR,new_data)))

##################
######Train Models Using automl
##################
#automl is an automated machine learning function found in the H2O package
#auto ML, creating multiple models
regression.auto_inc <- h2o.automl( y = y.dep, x = x.indep, training_frame = train.h2o, max_models=10,nfolds=5,stopping_metric = "MAE",stopping_rounds=15,sort_metric="MAE")
#View the AutoML Leader
regression.auto_inc@leader #this is the leading model using MAE as the sorting metric
#look at all of the models in the leaderboard
lb<-regression.auto_inc@leaderboard
print(lb, n = nrow(lb))  # Print all rows instead of default (6 rows)
#look at all of the models in the leaderboard, adding extra columns: training time, prediction time, algo type
lb2<-h2o.get_leaderboard(object = regression.auto_inc, extra_columns = 'ALL')
lb2

#predict using the leading model
autopred <- h2o.predict(object=regression.auto_inc@leader,newdata=test.h2o)
autopred

#given a trained h2o model, compute the leading model performance on the test dataset
autoperf <- h2o.performance(regression.auto_inc@leader,test.h2o)
autoperf
h2o.mae(autoperf)
h2o.rmse(autoperf)
h2o.r2(autoperf)

#residual analysis
h2o.explain(regression.auto_inc@leader,test.h2o)

##################
#####Train New  Error Prediction Model
##################

#using your data and the previous model, build a gbm model to predict the error in prediction from the above gbm model 
prederror = h2o.predict(object = proclength_gbm, newdata = your_data.h2o) #predict_proc_length
error_data<-your_data.h2o
error_data$error<-abs(prederror-your_data.h2o["Patient.In.Room.Duration"])

#break data into testing and training sets
trainIndex = sample(1:nrow(your_data), size = round(0.7*nrow(your_data)),replace=FALSE)
train = your_data[trainIndex ,]
test = your_data[-trainIndex ,] 
train.h2o<-as.h2o(train)
test.h2o<-as.h2o(test)
y.dep<-"error"
x.indep<-setdiff(names(train.h2o), y.dep)

error_gbm<-h2o.gbm(y=y.dep,x=x.indep,training_frame=train.h2o,stopping_rounds=3,stopping_metric = "MAE",stopping_tolerance=0.01,nfolds = 5,ntrees=500)
###saving error prediction model
h2o.saveModel(error_gbm, path = YOUR_PATH, force = TRUE) #force is to overwrite any existing model; note the new folder created
