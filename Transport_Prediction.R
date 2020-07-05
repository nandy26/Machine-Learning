#Mini Project-Predicting the mode of transport
rm(list=ls())
setwd("D:/BABI/Machine_Learning_M6/Mini_Project")
car_dataset=read.csv("Cars_edited.csv")
str(car_dataset)

#######################Data clensing ##############################
car_dataset$Engineer=as.factor(car_dataset$Engineer)
car_dataset$MBA=as.factor(car_dataset$MBA)
car_dataset$license=as.factor(car_dataset$license)
summary(car_dataset)
#Checking for missing values
sum(is.na(car_dataset))
#when MARGIN=1, it applies over rows, whereas with MARGIN=2, it works over columns. 
#Note that when you use the construct MARGIN=c(1,2), it applies to both rows and columns;
apply(car_dataset,2,function(x) sum(is.na(x)))
car_dataset=na.omit(car_dataset)#Remove the NA value
#Outliers
boxplot(car_dataset$Age,col="gray",main="Age")
boxplot(car_dataset$Work.Exp,col="gray",main="Working Experience")
boxplot(car_dataset$Salary,col="gray",main="Salary")
boxplot(car_dataset$Distance,col="gray",main="Distance")
#Uni-Variate Analysis-continous variable
par(mfrow=c(2,2))
hist(car_dataset$Age,col="blue",main="Employees Age")
hist(car_dataset$Work.Exp,col="blue",main="Employees Experience")
hist(car_dataset$Salary,col="blue",main="Employees Salary")
hist(car_dataset$Distance,col="blue",main="Travelling Distance")
#categorical variable
plot(car_dataset$Gender,main="Employees Gender")
plot(car_dataset$Engineer,main="Engineering Graduates")
plot(car_dataset$MBA,main="MBA Graduates")
plot(car_dataset$license,main="License")
qplot(car_dataset$Transport,main="Transport mode of employees")

#Bivariate analaysis
ggplot(car_dataset, aes(Age, ..count.., fill = Transport)) + geom_bar(position="dodge")
ggplot(car_dataset, aes(Work.Exp, ..count.., fill = Transport)) + geom_bar(position="dodge")
ggplot(car_dataset, aes(Salary, ..count.., fill = Transport)) + geom_bar(position="dodge")
ggplot(car_dataset, aes(Distance, ..count.., fill = Transport)) + geom_bar(position="dodge")
# contingency table of dicotomous variables with target variable
cat_data=subset(car_dataset,select=c(Gender,Engineer,MBA,license))#variable licence may be important one
par(mfrow=c(2,2))
for (i in names(cat_data))
{
  print(i)
  print(table(car_dataset$Transport,cat_data[[i]]))
  barplot(table(car_dataset$Transport,cat_data[[i]]),
          col=c("gray","red"),main=i)
}
dev.off()#To close/remove the plots 
## Check visual association pattern for continous predictor variables
library(GGally)
ggpairs(car_dataset[, c("Age","Work.Exp",
                        "Salary","Distance")],ggplot2::aes(colour = as.factor(car_dataset$Transport)))

###############Multi-collinearity############################
#Correlation between the continous variables
library(corrplot)
Cor_Matrix=cor(car_dataset[,c(1,5,6,7)])#Age, salary and work.Exp are highly correlated
corrplot(Cor_Matrix)
####################Data split##########################
library(caTools)
set.seed(1234)
car_dataset$Transport=ifelse(car_dataset$Transport=="Car",1,0)
car_dataset$Transport=as.factor(car_dataset$Transport)
sample=sample.split(car_dataset,SplitRatio=0.7)
car_dataset_train=subset(car_dataset,sample==TRUE)
car_dataset_test=subset(car_dataset,sample==FALSE)
car_dataset_test_smote=car_dataset_test
########################Data Preparation -SMOTE#########################
round(sum(car_dataset_train$Transport==1)/nrow(car_dataset_train),4)#14% use car as a mode of transport
library(DMwR)#DMwR-Data Mining with R
table(car_dataset_train$Transport)
Smote_data=SMOTE(car_dataset_train$Transport~.,car_dataset_train,perc.over = 53,perc.under =1000,k=3)
#After performing SMOTE
round(sum(Smote_data$Transport==1)/nrow(Smote_data),4)
table(Smote_data$Transport)


#################Logistic Regression without SMOTE################################
log_model_1=glm(car_dataset_train$Transport~.,data=car_dataset_train,family="binomial")
summary(log_model_1)
library(car)
vif(log_model_1)
log_model_2=glm(car_dataset_train$Transport~Gender+Engineer+MBA+Salary+Distance+license,data=car_dataset_train,family="binomial")
summary(log_model_2)
vif(log_model_2)
log_model_3=glm(car_dataset_train$Transport~Salary+Distance+license,data=car_dataset_train,family="binomial")
summary(log_model_3)
vif(log_model_3)#significant variables-salary,distance and license
exp(log_model_3$coefficients)#Odds ratio
#Testing data
round(sum(car_dataset_test$Transport==1)/nrow(car_dataset_test),3)
table(car_dataset_test$Transport)
car_dataset_test$prob_test=predict(log_model_3,newdata=car_dataset_test,type="response")
car_dataset_test$Predict_result=ifelse(car_dataset_test$prob_test>0.5,1,0)
car_dataset_test$Predict_result=as.factor(car_dataset_test$Predict_result)

#####################Performance Metrics for test data#################################
library(caret)
library(dplyr)
library(purrr)
confusionMatrix(car_dataset_test$Predict_result,car_dataset_test$Transport,positive="1")
Confusion_matrix_lg=table(car_dataset_test$Predict_result,car_dataset_test$Transport)
#ROC plot
library(ROCR)
ROC_pred=prediction(car_dataset_test$prob_test,car_dataset_test$Transport)
AUC_lg_array=performance(ROC_pred,"auc")#AUC >0.5 , so it is predicting 0 as 0 and 1 as 1
AUC_lg=round(as.numeric(AUC_lg_array@y.values),3)
ROC_plot=performance(ROC_pred,"tpr","fpr")
plot(ROC_plot,colorize=TRUE,main="ROC Curve",print.cutoffs.at=seq(0,1,by=0.1))
KS_lg=round(max(ROC_plot@y.values[[1]]-ROC_plot@x.values[[1]]),3)#KS chart
library(ineq)
Gini_lg=round(ineq(car_dataset_test$Predict_result,type="Gini"),3)#Gini

###################Logistic Regression with SMOTE data#########################
log_model_smote=glm(Smote_data$Transport~Salary+Distance+license,data=Smote_data,family="binomial")
summary(log_model_smote)
vif(log_model_smote)
car_dataset_test_smote$prob_test=predict(log_model_smote,newdata=car_dataset_test_smote,type="response")
car_dataset_test_smote$Predict_result=ifelse(car_dataset_test_smote$prob_test>0.5,1,0)
car_dataset_test_smote$Predict_result=as.factor(car_dataset_test_smote$Predict_result)
confusionMatrix(car_dataset_test_smote$Predict_result,car_dataset_test_smote$Transport,positive="1")

##################KNN-K Nearest Neighbours ##########################
library(class)
car_dataset_Knn=car_dataset
sample_KNN=sample.split(car_dataset_Knn,SplitRatio=0.7)
car_dataset_Knn$Age=as.numeric(car_dataset_Knn$Age)
car_dataset_Knn$Gender=as.numeric(car_dataset_Knn$Gender)
car_dataset_Knn$Engineer=as.numeric(car_dataset_Knn$Engineer)
car_dataset_Knn$MBA=as.numeric(car_dataset_Knn$MBA)
car_dataset_Knn$Work.Exp=as.numeric(car_dataset_Knn$Work.Exp)
car_dataset_Knn$Salary=as.numeric(car_dataset_Knn$Salary)
car_dataset_Knn$Distance=as.numeric(car_dataset_Knn$Distance)
car_dataset_Knn$license=as.numeric(car_dataset_Knn$license)
car_dataset_train_KNN=subset(car_dataset_Knn,sample_KNN==TRUE)
car_dataset_test_KNN=subset(car_dataset_Knn,sample_KNN==FALSE)
smote_knn=Smote_data
test=subset(car_dataset_Knn,sample_KNN==FALSE)

car_dataset_test_KNN$Status=knn(car_dataset_train_KNN[,-c(9)],car_dataset_test_KNN[,-c(9)],car_dataset_train_KNN$Transport,k=3)
confusionMatrix(car_dataset_test_KNN$Status,car_dataset_test_KNN$Transport,positive="1")
Confusion_matrix_KNN=table(car_dataset_test_KNN$Status,car_dataset_test_KNN$Transport)

##with smote
#test$Status=knn(smote_knn[,-c(9)],test[,-c(9)],smote_knn$Transport,k=3)
#confusionMatrix(test$Status,test$Transport,positive="1")#accuracy comes down from 97 to 92%

###################Ensemble methods##############################################
##Bagging

library(rpart)
car_dataset_bag=car_dataset
sample_bag=sample.split(car_dataset_bag,SplitRatio = 0.7)
car_dataset_bag_train=subset(car_dataset_bag,sample_bag==TRUE)
car_dataset_bag_test=subset(car_dataset_bag,sample_bag==FALSE)
bag_model=bagging(car_dataset_bag_train$Transport~.,data=car_dataset_bag_train,
                  control=rpart.control(maxdepth = 5,minsplit = 4))

car_dataset_bag_test$Status=predict(bag_model,car_dataset_bag_test)
confusionMatrix(car_dataset_bag_test$Status,car_dataset_bag_test$Transport,positive="1")


##Boosting
library(gbm)
car_dataset_boost=car_dataset
sample_boost=sample.split(car_dataset_boost,SplitRatio = 0.7)
car_dataset_boost_train=subset(car_dataset_boost,sample_boost==TRUE)
car_dataset_boost_test=subset(car_dataset_boost,sample_boost==FALSE)
car_dataset_boost_train$Transport=as.character(car_dataset_boost_train$Transport)
boost_model=gbm(car_dataset_boost_train$Transport~.,data=car_dataset_boost_train,distribution = 'bernoulli',n.trees =3000,interaction.depth = 4,shrinkage=0.01)
summary(boost_model)
car_dataset_boost_test$prob_result=predict(boost_model,car_dataset_boost_test,type="response",n.trees = 3000)
car_dataset_boost_test$Status=ifelse(car_dataset_boost_test$prob_result>0.5,1,0)
car_dataset_boost_test$Transport=as.factor(car_dataset_boost_test$Transport)
car_dataset_boost_test$Status=as.factor(car_dataset_boost_test$Status)
confusionMatrix(car_dataset_boost_test$Status,car_dataset_boost_test$Transport,positive="1")
