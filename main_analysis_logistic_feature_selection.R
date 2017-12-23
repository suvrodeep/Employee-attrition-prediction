library(caret)
library(ggplot2)
library(dummies)
library(pROC)
library(randomForest)
library(Hmisc)
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

#Exploring collinearity
dum.colnames <- c("department", "salary")
df.train <- dummy.data.frame(data = df.train, names = dum.colnames, sep = "_")
df.test <- dummy.data.frame(data = df.test, names = dum.colnames, sep = "_")

cor.matrix <- Hmisc::rcorr(as.matrix(df.train))
print(cor.matrix)

#FEATURE SELECTION
#Rank features by importance
#Without dummy variables

set.seed(123457)
train_index <- caret::createDataPartition(df$department, p = 0.7, list = FALSE)
df.train <- df[train_index,]
df.test <- df[-train_index,]

control <- caret::trainControl(method = "repeatedcv", number = 10, repeats = 3)
model <- caret::train(left~., data = df.train, method = "glm", trControl = control)
importance <- caret::varImp(model)
plot(importance)
imp <- setDT(importance$importance, keep.rownames = TRUE)
colnames(imp) <- c("Feature", "Importance")
imp <- imp[order(imp$Importance, decreasing = TRUE),]
print(imp)

fit2 <- glm(left ~ satisfaction_level + time_spend_company + Work_accident + salary + number_project 
            + average_montly_hours, data = df.train, family = "binomial")
df.train$predicted_prob <- predict(fit2, df.train, type = "response")
df.test$predicted_prob <- predict(fit2, df.test, type = "response")

#Plot cutoff vs accuracy
accPlot.dataFrame <- data.frame(CUTOFF = seq(0, 1, length = 10000), ACCURACY = numeric(10000))

#Plot for training data set
for (index in 1:10000) {
  pred <- ifelse((df.train$predicted_prob > accPlot.dataFrame[index, 1]), 1, 0)
  true.positives <- sum(pred == 1 & df.train$left == 1)
  true.negatives <- sum(pred == 0 & df.train$left == 0)
  accPlot.dataFrame$ACCURACY[index] <- ((true.positives + true.negatives) / length(df.train$left)) * 100
}
ggplot2::ggplot(data = accPlot.dataFrame, mapping = aes(x = CUTOFF, y = ACCURACY, col)) + geom_line(size = 1)
idealCutoff <- accPlot.dataFrame$CUTOFF[which.max(accPlot.dataFrame$ACCURACY)]
acc.max <- accPlot.dataFrame$ACCURACY[which.max(accPlot.dataFrame$ACCURACY)]

df.train$predicted_outcome <- ifelse((df.train$predicted_prob > idealCutoff), 1, 0)
df.test$predicted_outcome <- ifelse((df.test$predicted_prob > idealCutoff), 1, 0)

#Confusion matrices with optimal cutoff
conf.matrix.train <- caret::confusionMatrix(df.train$predicted_outcome, df.train$left)
conf.matrix.train

conf.matrix.test <- caret::confusionMatrix(df.test$predicted_outcome, df.test$left)
conf.matrix.test

#ROC for Logistic regerssion with selected predictors without dummy variables
roc.train <- pROC::roc(df.train$left, df.train$predicted_prob)
roc.test1 <- pROC::roc(df.test$left, df.test$predicted_prob)
pROC::plot.roc(roc.train, col = "BLUE")
pROC::plot.roc(roc.test1, add = TRUE, col = "GREEN")


#With dummy variables
set.seed(123457)
train_index <- caret::createDataPartition(df$department, p = 0.7, list = FALSE)
df.train <- df[train_index,]
df.test <- df[-train_index,]
dum.colnames <- c("department", "salary")
df.train <- dummy.data.frame(data = df.train, names = dum.colnames, sep = "_")
df.test <- dummy.data.frame(data = df.test, names = dum.colnames, sep = "_")

control <- caret::trainControl(method = "repeatedcv", number = 10, repeats = 3)
model <- caret::train(left~., data = df.train, method = "glm", trControl = control)
importance <- caret::varImp(model)
plot(importance)
imp <- setDT(importance$importance, keep.rownames = TRUE)
colnames(imp) <- c("Feature", "Importance")
imp <- imp[order(imp$Importance, decreasing = TRUE),]
print(imp)

fit3 <- glm(left ~ satisfaction_level + time_spend_company + Work_accident + salary_low + number_project + salary_high 
            + average_montly_hours, data = df.train, family = "binomial")
df.train$predicted_prob <- predict(fit3, df.train, type = "response")
df.test$predicted_prob <- predict(fit3, df.test, type = "response")

#Plot cutoff vs accuracy
#Plot for training data set
for (index in 1:10000) {
  pred <- ifelse((df.train$predicted_prob > accPlot.dataFrame[index, 1]), 1, 0)
  true.positives <- sum(pred == 1 & df.train$left == 1)
  true.negatives <- sum(pred == 0 & df.train$left == 0)
  accPlot.dataFrame$ACCURACY[index] <- ((true.positives + true.negatives) / length(df.train$left)) * 100
}
ggplot2::ggplot(data = accPlot.dataFrame, mapping = aes(x = CUTOFF, y = ACCURACY, col)) + geom_line(size = 1)
idealCutoff <- accPlot.dataFrame$CUTOFF[which.max(accPlot.dataFrame$ACCURACY)]
acc.max <- accPlot.dataFrame$ACCURACY[which.max(accPlot.dataFrame$ACCURACY)]

df.train$predicted_outcome <- ifelse((df.train$predicted_prob > idealCutoff), 1, 0)
df.test$predicted_outcome <- ifelse((df.test$predicted_prob > idealCutoff), 1, 0)

#Confusion matrices with optimal cutoff
conf.matrix.train <- caret::confusionMatrix(df.train$predicted_outcome, df.train$left)
conf.matrix.train

conf.matrix.test <- caret::confusionMatrix(df.test$predicted_outcome, df.test$left)
conf.matrix.test

#ROC for Logistic regerssion with selected predictors with dummy variables
roc.train <- pROC::roc(df.train$left, df.train$predicted_prob)
roc.test1 <- pROC::roc(df.test$left, df.test$predicted_prob)
pROC::plot.roc(roc.train, col = "BLUE")
pROC::plot.roc(roc.test1, add = TRUE, col = "GREEN")

