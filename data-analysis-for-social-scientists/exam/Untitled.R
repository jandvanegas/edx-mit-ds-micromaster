# Preliminaries
#-------------------------------------------------
#install.packages('perm')
library(perm)
library(modelr)
library(np)
library(tidyverse)
rm(list = ls())
setwd("")
getwd()

perms <- chooseMatrix(6,3)
A <- matrix(c(36.4, 38.2, 37.1, 37.3, 36, 37.6), ncol=1, byrow=TRUE)
treatment_avg <- (1/3)*perms%*%A
control_avg <- (1/3)*(1-perms)%*%A
test_statistic <- abs(treatment_avg-control_avg)
real_effect <- (abs(38.2 + 37.6 + 37.1 - 36.4 - 37.3 -36))/3
sum(test_statistic >= real_effect)/nrow(perms)
nrow(perms)




