# install packages
install.packages(c("titanic","dplyr", "tidyverse"))
library(titanic)
library(dplyr)
library(tidyverse)

head(titanic_train)

## Droup NA (missing values)
titanic_train <- na.omit(titanic_train)
nrow(titanic_train)

## SPLIT DATA
set.seed(10)
n <- nrow(titanic_train)
id <- sample(1:n, size = n*0.7)
train_data <- titanic_train[id, ]
test_data <- titanic_train[-id, ]



## Train Model
logit_model <- glm(Survived ~ Pclass + Age , data = train_data, 
                   family = "binomial")
p_train <- predict(logit_model, type ="response")
train_data$pred <- ifelse(p_train >= 0.5, 0,1)
mean(train_data$Survived == train_data$pred)


## Test Model
p_test <- predict(logit_model,newdata = test_data, type ="response")
test_data$pred <- ifelse(p_test >= 0.5, 0,1)
mean(test_data$Survived == test_data$pred)


## Confusion Matrix
conM <- table(test_data$pred,test_data$Sex, 
              dnn = c("Predicted","Actual"))


## Model Evaluation
Acc <-  (conM[1,1] + conM[2,2]) / sum(conM)
Prec <-  conM[2,2]/ (conM[2,1] + conM[2,2])
Recall <- conM[2,2]/ (conM[1,2] + conM[2,2])
F1 <- 2*Prec*Recall/(Prec+Recall)
cat("Result:",
    "\n","Accuracy:",Acc,
    "\n","Precision:",Prec,
    "\n","Recall:",Recall,
    "\n","F1:",F1)
