library(caret)
library(ggplot2)
library(dummies)
library(pROC)
library(randomForest)
library(data.table)

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

df.train$predicted_outcome <- predict(model.rf, newdata = df.train, type = "response")
prob <- predict(model.rf, newdata = df.train, type = "prob")
df.train$predicted_prob_0 <- prob[,1]
df.train$predicted_prob_1 <- prob[,2]

df.test$predicted_outcome <- predict(model.rf, newdata = df.test, type = "response")
prob <- predict(model.rf, newdata = df.test, type = "prob")
df.test$predicted_prob_0 <- prob[,1]
df.test$predicted_prob_1 <- prob[,2]

#Confusion matrices
conf.matrix.train <- caret::confusionMatrix(df.train$predicted_outcome, df.train$left)
conf.matrix.train

conf.matrix.test <- caret::confusionMatrix(df.test$predicted_outcome, df.test$left)
conf.matrix.test

#ROC curve with all pedictors
roc.train <- pROC::roc(df.train$left, df.train$predicted_prob_1)
roc.test1 <- pROC::roc(df.test$left, df.test$predicted_prob_1)
pROC::plot.roc(roc.train, col = "BLUE")
pROC::plot.roc(roc.test1, add = TRUE, col = "GREEN")


#Feature selection in Random Forest (Recursive Feature Eliminaion)
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

imp <- setDT(data.frame(vars.imp), keep.rownames = TRUE)
colnames(imp) <- c("Feature", "Mean_Decrease_Gini")
imp <- imp[order(imp$Mean_Decrease_Gini, decreasing = TRUE),]
print(imp)
ggplot(data = imp, mapping = aes(x = Mean_Decrease_Gini, y = reorder(Feature, Mean_Decrease_Gini))) + geom_point()


model.rf <- randomForest::randomForest(left ~ satisfaction_level + number_project + average_montly_hours + time_spend_company 
                                       + last_evaluation + department, data = df.train, verbose = TRUE)
df.train$predicted_outcome <- predict(model.rf, newdata = df.train, type = "response")
df.test$predicted_outcome <- predict(model.rf, newdata = df.test, type = "response")

prob <- predict(model.rf, newdata = df.train, type = "prob")
df.train$predicted_prob_0 <- prob[,1]
df.train$predicted_prob_1 <- prob[,2]

prob <- predict(model.rf, newdata = df.test, type = "prob")
df.test$predicted_prob_0 <- prob[,1]
df.test$predicted_prob_1 <- prob[,2]

#Confusion matrices
conf.matrix.train <- caret::confusionMatrix(df.train$predicted_outcome, df.train$left)
conf.matrix.train

conf.matrix.test <- caret::confusionMatrix(df.test$predicted_outcome, df.test$left)
conf.matrix.test


#ROC curve with feature selected predictors
roc.train <- pROC::roc(df.train$left, df.train$predicted_prob_1)
roc.test1 <- pROC::roc(df.test$left, df.test$predicted_prob_1)
pROC::plot.roc(roc.train, col = "BLUE")
pROC::plot.roc(roc.test1, add = TRUE, col = "GREEN")
