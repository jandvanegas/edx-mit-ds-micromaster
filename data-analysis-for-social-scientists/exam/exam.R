
# Preliminaries
#-------------------------------------------------
#install.packages('perm')
library(perm)
library(modelr)
library(np)
library(tidyverse)
rm(list = ls())

setwd("/Users/estefaniasosa/Documents/notes/edx-mit-ds-micromaster/data-analysis-for-social-scientists/exam")
getwd()

perms <- chooseMatrix(6,3)
A <- matrix(c(36.4, 38.2, 37.1, 37.3, 36, 37.6), ncol=1, byrow=TRUE)
treatment_avg <- (1/3)*perms%*%A
control_avg <- (1/3)*(1-perms)%*%A
test_statistic <- abs(treatment_avg-control_avg)
real_effect <- (abs(38.2 + 37.6 + 37.1 - 36.4 - 37.3 -36))/3
sum(test_statistic >= real_effect)/nrow(perms)
nrow(perms)

# .csv
treatment_control <- na.omit(read.csv("data_myData.csv"))
treatment_avg <- mean(treatment_control$Y * treatment_control$T)
control_avg <- mean(treatment_control$Y * (1 - treatment_control$T))
ate <- treatment_avg - control_avg
nc <- sum(treatment_control$T)
nt <- sum(1 - treatment_control$T)

treatment_avg_sq <- sum((treatment_control$Y * treatment_control$T - treatment_avg) ^ 2)/(nt-1)
control_avg_sq <- sum((treatment_control$Y * (1 - treatment_control$T) - control_avg) ^2)/(nc -1)
v_neiman <- treatment_avg_sq/nt + control_avg_sq/nc
t_critical <- abs(qt(0.05/2, df = 99))
interval_a <- ate - t_critical * sqrt(v_neiman)
interval_b <- ate + t_critical * sqrt(v_neiman)



