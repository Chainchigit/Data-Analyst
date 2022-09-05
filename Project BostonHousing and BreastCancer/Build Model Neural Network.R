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

# 3. Build Model Neural Network
## Droup NA (missing values)
head(BreastCancer_Dna)
    
BreastCancer_Dna <- na.omit(BreastCancer)
nrow(BreastCancer_Dna)
glimpse(BreastCancer_Dna)    

# train test split
set.seed(2)
n <- nrow(BreastCancer_Dna)
id <- sample(1:n3,size = n3*0.7)

BreastCancer_Dna_Train <- BreastCancer_Dna[id, -1 ]
BreastCancer_Dna_Test <-  BreastCancer_Dna[-id, ]

# model training
nn_model <- nnet(Class ~ .,
                 data = BreastCancer_Dna_Train,
                 size = 3)

# plot networks
plotnet(nn_model)

# model evaluation
p <- predict(nn_model, newdata = BreastCancer_Dna_Test, type = "class")
mean(p == BreastCancer_Dna_Test$Class)
