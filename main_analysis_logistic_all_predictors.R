library(caret)
library(ggplot2)
library(dummies)
library(pROC)
library(randomForest)


#Importing dataset
df <- read.csv("HR_data.csv", header = TRUE)

#Renaming columns and cleaning the dataset
colnames(df)[colnames(df) == "sales"] <- "department"

#Creating factors and partitions(based on department)
df$department <- as.factor(df$department)
df$salary <- as.factor(df$salary)
df$left <- as.factor(df$left)

set.seed(123457)
train_index <- caret::createDataPartition(df$department, p = 0.7, list = FALSE)
df.train <- df[train_index,]
df.test <- df[-train_index,]


#Logistic regression without feature selection
fit1 <- glm(left ~ ., data = df.train, family = "binomial")
df.train$predicted_prob <- predict(fit1, df.train, type = "response")
df.test$predicted_prob <- predict(fit1, df.test, type = "response")

#Classification with cutoff=0.5
df.train$predicted_outcome <- ifelse((df.train$predicted_prob > 0.5), 1, 0)
conf.matrix.train <- caret::confusionMatrix(df.train$predicted_outcome, df.train$left)
conf.matrix.train

df.test$predicted_outcome <- ifelse((df.test$predicted_prob > 0.5), 1, 0)
conf.matrix.test <- caret::confusionMatrix(df.test$predicted_outcome, df.test$left)
conf.matrix.test

#ROC for Logistic regression with all predictors and 0.5 as cutoff
roc.train <- pROC::roc(df.train$left, df.train$predicted_outcome)
pROC::plot.roc(roc.train)

roc.test1 <- pROC::roc(df.test$left, df.test$predicted_outcome)
pROC::plot.roc(roc.test1)


#Determining optimal cutoff
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

#ROC for Logistic regerssion with all predictors with optimal cutoff
roc.train <- pROC::roc(df.train$left, df.train$predicted_outcome)
pROC::plot.roc(roc.train)

roc.test1 <- pROC::roc(df.test$left, df.test$predicted_outcome)
pROC::plot.roc(roc.test1)


