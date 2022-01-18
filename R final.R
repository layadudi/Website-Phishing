Data = read.csv("Data.csv")
summary(Data)
csv_result.old = read.csv("csv_result-old.csv")
new_data = rbind(Data,csv_result.old)
new_data$Result = as.factor(new_data$Result)
summary(new_data)
sum(is.na(new_data))

library(ROSE)
balanced.data = ovun.sample(Result ~., data = new_data, ,method = "over", N =14000)$data

set.seed(32)
indx <- sample(2, nrow(balanced.data), replace = TRUE, prob = c(0.7,0.3))
train_data <- balanced.data[indx == 1,]
test_data <- balanced.data[indx == 2,]

## Decision Tree
library(rpart)
library(rpart.plot)
dt_new = rpart(Result~., data= new_data, control=rpart.control(minsplit=50, minbucket=100, cp=0))
rpart.plot(dt_new)
data_predict = predict(dt_new, test_data, type = "class")
mean(test_data$Result == data_predict)
library(caret)
confusionMatrix(data_predict,test_data$Result)

## Naive Bayes
library(e1071)
naive_e1071 <- naiveBayes(Result ~ ., data = train_data)

predict(naive_e1071, newdata = test_data, type = "raw")
preded <- predict(naive_e1071, newdata = test_data)
table(preded, test_data$Result)
mean(preded == test_data$Result)
confusionMatrix(preded,test_data$Result)

## Random Forest Model
set.seed(32)
#install.packages("randomForest")
library(randomForest)
rf <- randomForest(Result ~ ., data = train_data, mtry =sqrt(ncol(train_data)-1) , ntree = 100, proximity = T, importance = T)
rf$predicted
library(caret)
confusionMatrix(rf$predicted,test_data$Result)

#########################CROSS VALIDATION on DT##################

set.seed(32)
k <- 10
nmethod <- 1
folds <- cut(seq(1,nrow(new_data)),breaks=k,labels=FALSE) 
model.err <- matrix(-1,k,nmethod,dimnames=list(paste0("Fold", 1:k), c("DT")))
library(rpart)

for(i in 1:k)
{ 
  testindexes <- which(folds==i, arr.ind=TRUE) 
  test <- new_data[testindexes, ] 
  train <- new_data[-testindexes, ] 
  
  
  dt_model = rpart(Result~., data = train, control=rpart.control(minsplit=200, minbucket=100, cp=0.01))
  data_predict = predict(dt_model, test, type = "class")
  model.err[i] <- mean(test$Result != data_predict)
  
  
}

mean(model.err)
result = confusionMatrix(data_predict, test$Result, mode = "prec_recall")
result
result$byClass["Precision"]
result$byClass["Recall"]
result$byClass["F1"]


#########################CROSS VALIDATION on NB##################

set.seed(32)
k <- 10
nmethod <- 1
folds <- cut(seq(1,nrow(new_data)),breaks=k,labels=FALSE) 
model.err <- matrix(-1,k,nmethod,dimnames=list(paste0("Fold", 1:k), c("NB")))

for(i in 1:k)
{ 
  testindexes <- which(folds==i, arr.ind=TRUE) 
  test <- new_data[testindexes, ] 
  train <- new_data[-testindexes, ] 
  
  
  library(e1071)
  naive_e1071 <- naiveBayes(train$Result ~ ., data = train)
  preded <- predict(naive_e1071, newdata = test)
  
  model.err[i] <- mean(test$Result != preded)
  
  
}

mean(model.err)
sensitivity(preded, test$Result)


result = confusionMatrix(preded, test$Result, mode = "prec_recall")
result
result$byClass["Precision"]
result$byClass["Recall"]
result$byClass["F1"]


#########################CROSS VALIDATION on RF##################
set.seed(32)
k <- 10
nmethod <- 1
folds <- cut(seq(1,nrow(train_data)),breaks=k,labels=FALSE) 
model.err <- matrix(-1,k,nmethod,dimnames=list(paste0("Fold", 1:k), c("RF")))
library(rpart)

for(i in 1:k)
{ 
  testindexes <- which(folds==i, arr.ind=TRUE) 
  test <- train_data[testindexes, ] 
  train <- train_data[-testindexes, ] 
  
  
  # Random Forest Model
  #install.packages("randomForest")
  library(randomForest)
  rf <- randomForest(Result ~ ., data = train, mtry = sqrt(ncol(Data)- 1), ntree = 100, proximity = T, importance = T)
  predictedrf = predict(rf, newdata = test, type="class")
  model.err[i] <- mean(test$Result != predictedrf)
  
  
}
mean(model.err)
sensitivity(predictedrf, test$Result)

result = confusionMatrix(predictedrf, test$Result, mode = "prec_recall")
result
result$byClass["Precision"]
result$byClass["Recall"]
result$byClass["F1"]

###################################################################################








