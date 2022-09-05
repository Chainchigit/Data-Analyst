## Install Packages
install.packages(c("tidyverse","caret","mlbench","readr","dplyr","MLmetrics"))
library(tidyverse)
library(caret)
library(mlbench)
library(readr)
library(dplyr)
library(MLmetrics)

## Data Transformation
churn <- churn %>%
    mutate(churn  = as.factor(churn),
           internationalplan  = as.factor(internationalplan),
           voicemailplan  = as.factor(voicemailplan)) 

## Miss value
mean(complete.cases(churn))

## Preview data
churn %>% head()

## 1. Split data
set.seed(42)
id <- createDataPartition(y = churn$churn,
                          p = 0.7,
                          list = FALSE)

train_df <- churn[id, ]
test_df <- churn[-id, ]

## 2. train model
## Logistic Regression
set.seed(42)

crtl <- trainControl(
    method = "CV",
    number = 3,
    verboseIter = TRUE
)

logistic_model <- train(churn ~ .,
                        data = train_df,
                        method ="glm",
                        trControl = crtl)

knn_model <- train(churn ~ .,
                   data = train_df,
                   method ="knn",
                   trControl = crtl)

rf_model <- train(churn ~ .,
                  data = train_df,
                  method ="rf",
                  trControl = crtl)

## Compere three models
result <- resamples(list(
    logisticReg = logistic_model,
    knn = knn_model,
    randomForest = rf_model
))
summary(result)

## test model
p <- predict(knn_model, newdata=test_df)

mean(p == test_df$churn)
table(p, test_df$churn, dnn =c("Prediction", "Actual"))

confusionMatrix(p,test_df$churn,
                positive = "Yes",
                mode = "prec_recall")
