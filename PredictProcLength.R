Supplemental Digital Content 7 -  R Code: Making Predictions Using Created Machine Learning Models
#########
#if you haven’t done this previously, download H2O
#https://docs.h2o.ai/h2o/latest-stable/h2o-docs/downloading.html
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

pkgs <- c("RCurl","jsonlite")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}

install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/latest_stable_R")))
library(h2o)
localH2O = h2o.init()
demo(h2o.kmeans)
#########

library(h2o)  #scalable open source machine learning platform
library(dplyr)
h2o.shutdown(prompt = TRUE) #shutting down H2O instance if there is an existing instance running

#load your dataset
setwd("C:/...") #set to location of your files, or use database connection through RODBC
new_data<-read.csv("atOR.csv")
#initialize H2O server
localH2O <- h2o.init(nthreads = -1)

#convert data to H2O format
new_data.h2o<-as.h2o(new_data)

#load preexisting model
proclength_gbm<-h2o.loadModel(MODEL_PATH) #set to location and name of model
error_gbm<-h2o.loadMode(MODEL_PATH) #set to location and name of error prediction model

#prediction of length and error, comparing to actual length and error from prediction, stored in a data frame 
pred_proclength<-as.data.frame(h2o.predict(proclength_gbm,new_data.h2o))
names(pred_proclength)[1]<-"pred_proclength"
pred_error<-as.data.frame(h2o.predict(error_gbm,new_data.h2o))
names(pred_error)[1]<-"pred_error"

#local explanation of any predicted output
row<-#your_choice_here
shapr_plot <- h2o.shap_explain_row_plot(proclength_gbm, new_data, row_index = row)

### load existing quantile error prediction models
q5OR<-h2o.loadModel(path = "C:/.../.../GBM_model_R_...") #file name is auto-chosen by H2O
q2OR<-h2o.loadModel(path = "C:/.../.../GBM_model_R_...") #file name is auto-chosen by H2O
q8OR<-h2o.loadModel(path = "C:/.../.../GBM_model_R_...") #file name is auto-chosen by H2O

#prediction interval based on quantile model
a<-h2o.predict(q5OR, your_data)
b<-h2o.predict(q2OR, your_data)
c<-h2o.predict(q8OR, your_data)

#quantile prediction interval output data as a table
quantile_prediction_table->(cbind(a,b,c))

#local explanation of any predicted output
row<-#your_choice_here
shapr_plot <- h2o.shap_explain_row_plot(q5OR, new_data, row_index = row)
