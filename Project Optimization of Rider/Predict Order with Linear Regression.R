# install packages
install.packages(c("dplyr", "nnet", "NeuralNetTools"))
install.packages("mlbench")
library(mlbench)
library(tidyverse)
library(dplyr)
library(nnet)
library(NeuralNetTools)
library(readr)

Rider <- read_csv("Rider.csv")
View(Rider)


## 1. Build Model linear Regression
head(Rider)

lm_Rider <- lm(Total_Order ~ ., data = Rider)
Rider$UTC_Hour <- predict(lm_Rider)

## Train RMSE
squared_error <- (Rider$UTC_Hour - Rider$Total_Order) ** 2
(rmse <- sqrt(mean(squared_error)) )

## split Data
set.seed(99)
n <- nrow(Rider)
id <- sample(1:n, size = n*0.7)
train_data <- Rider[id, ]
test_data <- Rider[-id, ] 

## Train Model
model_Rider <- lm(Total_Order ~ UTC_DATE  , data = train_data)
p_train <- predict(model_Rider)
error_train <- train_data$UTC_Hour - p_train
(rmse_train <- sqrt(mean((error_train) ** 2)))

## Test Model
p_test <- predict(model_Rider, newdata = test_data)
error_test <- test_data$UTC_Hour - p_test
(rmse_test <- sqrt(mean(error_test** 2)))

## Print Result
cat("Result:",
    "\n","RMSE Train:" ,rmse_train,
    "\n","RMSE Test:", rmse_test)
-----------------------------#Result--------------------------------
# Result: 
# RMSE Train: 1.710138 
# RMSE Test: 1.662732
