# install packages
install.packages(c("dplyr", "nnet", "NeuralNetTools"))
install.packages("mlbench")
library(mlbench)
library(tidyverse)
library(dplyr)
library(nnet)
library(NeuralNetTools)
data("BreastCancer")
data("BostonHousing")


## 1. Build Model linear Regression
head(BostonHousing)

lm_BH <- lm(medv ~ ., data = BostonHousing)
BostonHousing$predicted <- predict(lm_BH)

## Train RMSE
squared_error <- (BostonHousing$medv - BostonHousing$predicted) ** 2
(rmse <- sqrt(mean(squared_error)) )

## split Data
set.seed(18)
n <- nrow(BostonHousing)
id <- sample(1:n, size = n*0.7)
train_data <- BostonHousing[id, ]
test_data <- BostonHousing[-id, ] 

## Train Model
model_BH <- lm(medv ~ age + ptratio, data = train_data)
p_train <- predict(model_BH)
error_train <- train_data$medv - p_train
(rmse_train <- sqrt(mean((error_train) ** 2)))

## Test Model
p_test <- predict(model_BH, newdata = test_data)
error_test <- test_data$medv - p_test
(rmse_test <- sqrt(mean(error_test** 2)))

## Print Result
cat("Result:",
    "\n","RMSE Train:" ,rmse_train,
    "\n","RMSE Test:", rmse_test)
