library(caret)
library(ggplot2)
library(dummies)
library(pROC)
library(randomForest)

#Importing dataset
df <- read.csv("HR_data.csv", header = TRUE)

#Renaming columns and cleaning the dataset
colnames(df)[colnames(df) == "sales"] <- "department"

#Creating factors and partitions
df$department <- as.factor(df$department)
df$salary <- as.factor(df$salary)
df$left <- as.factor(df$left)

set.seed(123457)
train_index <- caret::createDataPartition(df$department, p = 0.7, list = FALSE)
df.train <- df[train_index,]
df.test <- df[-train_index,]

#IMPROVING THE MODEL

#Random Forests
model.rf <- randomForest::randomForest(left ~ ., data = df.train, verbose = TRUE)
df.train$predicted_outcome <- predict(model.rf, newdata = df.train)
df.test$predicted_outcome <- predict(model.rf, newdata = df.test)

#Confusion matrices
conf.matrix.train <- caret::confusionMatrix(df.train$predicted_outcome, df.train$left)
conf.matrix.train

conf.matrix.test <- caret::confusionMatrix(df.test$predicted_outcome, df.test$left)
conf.matrix.test

#Feature selection in Random Forest (Backward Selection)
set.seed(123457)
train_index <- caret::createDataPartition(df$department, p = 0.7, list = FALSE)
df.train <- df[train_index,]
df.test <- df[-train_index,]

control <- caret::rfeControl(functions = rfFuncs, method = "cv", number = 10)
results <- caret::rfe(df.train[,-(which(colnames(df.train) == "left"))], df.train$left, sizes = c(1:9), rfeControl = control, verbose = TRUE)
print(results)
ggplot(data = results) + geom_line()
vars.imp <- randomForest::importance(results$fit)
print(vars.imp)

model.rf <- randomForest::randomForest(left ~ satisfaction_level + number_project + average_montly_hours + time_spend_company 
                                       + last_evaluation + department, data = df.train, verbose = TRUE)
df.train$predicted_outcome <- predict(model.rf, newdata = df.train)
df.test$predicted_outcome <- predict(model.rf, newdata = df.test)

#Confusion matrices
conf.matrix.train <- caret::confusionMatrix(df.train$predicted_outcome, df.train$left)
conf.matrix.train

conf.matrix.test <- caret::confusionMatrix(df.test$predicted_outcome, df.test$left)
conf.matrix.test
