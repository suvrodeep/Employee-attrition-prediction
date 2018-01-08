library(caret)
library(ggplot2)
library(dummies)
library(pROC)
library(randomForest)
library(data.table)
library(dplyr)
library(klaR)

#Importing dataset
df <- read.csv("HR_data.csv", header = TRUE)

#Renaming columns and cleaning the dataset
colnames(df)[colnames(df) == "sales"] <- "department"

#Creating factors and partitions
df$department <- as.factor(df$department)
df$salary <- as.factor(df$salary)
df$left <- as.factor(df$left)


#k-means clustering
dummy.cols <- c("department", "salary")
df.km <- dummy.data.frame(df, name = dummy.cols, sep = "_")

set.seed(123457)
fit.kmeans <- kmeans(df.km[, -7], 5, nstart = 20)
bss <- fit.kmeans$betweenss/fit.kmeans$totss*100
wss <- fit.kmeans$withinss/fit.kmeans$totss*100
df.km$cluster <- fit.kmeans$cluster
results <- data.frame(1:5, 1, 1, 1, 1, 1)
colnames(results) <- c("Cluster", "Left", "Not Left", "Total", "Percentage Left", "Within_SS/Total_SS")

for (index in 1:5) {
  results[index, 2] <- sum(df.km$cluster == index & df.km$left == 1)
  results[index, 3] <- sum(df.km$cluster == index & df.km$left == 0)
  results[index, 4] <- results[index, 2] + results[index, 3]
  results[index, 5] <- (results[index, 2] * 100) / results[index, 4]
  results[index, 6] <- wss[index]
}

ggplot(data = results, mapping = aes(x = results$Cluster, y = results$`Percentage Left`)) + xlab("Cluster") + 
  ylab("Percentage") + geom_col()


#k-means with scaling
df.scale <- df[, c(1:6, 8)]
df.scale <- scale(df.scale)
df.merge1 <- df[, -c(1:6, 8)]
df.scale <- data.frame(df.scale)
df.km <- cbind(df.scale, df.merge1)

dummy.cols <- c("department", "salary")
df.km <- dummy.data.frame(df.km, name = dummy.cols, sep = "_")

set.seed(123457)
fit.kmeans <- kmeans(df.km[, -8], 5, nstart = 20)
bss <- fit.kmeans$betweenss/fit.kmeans$totss*100
wss <- fit.kmeans$withinss/fit.kmeans$totss*100
df.km$cluster <- fit.kmeans$cluster
results <- data.frame(1:5, 1, 1, 1, 1, 1)
colnames(results) <- c("Cluster", "Left", "Not Left", "Total", "Percentage Left", "Within_SS/Total_SS")

for (index in 1:5) {
  results[index, 2] <- sum(df.km$cluster == index & df.km$left == 1)
  results[index, 3] <- sum(df.km$cluster == index & df.km$left == 0)
  results[index, 4] <- results[index, 2] + results[index, 3]
  results[index, 5] <- (results[index, 2] * 100) / results[index, 4]
  results[index, 6] <- wss[index]
  
}

ggplot(data = results, mapping = aes(x = results$Cluster, y = results$`Percentage Left`)) + xlab("Cluster") + 
  ylab("Percentage") + geom_col()


#k-modes clustering
set.seed(123457)
fit.kmodes <- klaR::kmodes(df.km[, -7], modes = 5, iter.max = 10)

df.km$cluster <- fit.kmodes$cluster
results <- data.frame(cluster = 1:5, 1, 1, 1, 1)
colnames(results) <- c("Cluster", "Left", "Not Left", "Total", "Percentage Left")

for (index in 1:5) {
  results[index, 2] <- sum(df.km$cluster == index & df.km$left == 1)
  results[index, 3] <- sum(df.km$cluster == index & df.km$left == 0)
  results[index, 4] <- results[index, 2] + results[index, 3]
  results[index, 5] <- (results[index, 2] * 100) / results[index, 4]
}

ggplot(data = results, mapping = aes(x = results$Cluster, y = results$`Percentage Left`)) + xlab("Cluster") + 
  ylab("Percentage") + geom_col()








