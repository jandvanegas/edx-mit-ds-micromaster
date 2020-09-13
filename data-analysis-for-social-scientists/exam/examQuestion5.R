#Exercise5

library(perm)
library(modelr)
library(np)
library(tidyverse)
library(car)
library(data.table)

rm(list = ls())
#install.packages('car')
#install.packages("data.table", dependencies=TRUE)

setwd("/Users/estefaniasosa/Documents/notes/edx-mit-ds-micromaster/data-analysis-for-social-scientists/exam")
getwd()

houseAndCrime <- read.csv("data_House_Prices_and_Crime_1.csv")
mena_homicides <- mean(houseAndCrime$Homicides)
seventyPercent <- quantile(houseAndCrime$Homicides, 0.75)
print(seventyPercent)
stdHomicides <- sd(houseAndCrime$Homicides, na.rm = FALSE)
medianHouse <- median(houseAndCrime$index_nsa)

index <- houseAndCrime$index_nsa
hom <- houseAndCrime$Homicides
rob <- houseAndCrime$Robberies
assa <- houseAndCrime$Assaults
print(hom)
multil <- lm(index ~ hom + rob + assa, data = houseAndCrime)
summary(multil)

multil2 <- lm(index_nsa ~ Homicides + Robberies + Assaults, data = houseAndCrime)
summary(multil2)
predict(multil2, houseAndCrime[1,])

prediction<-predict(multil2, houseAndCrime)
mean(abs(houseAndCrime$index_nsa - prediction))
confint(multil2)

multi3 <- lm(index_nsa ~ Homicides, data = houseAndCrime)
summary(multi3)

houseAndCrime$AssaHomRel <- assa-hom
houseAndCrime$RobHomRel <- rob-hom
multi4 <- lm(index_nsa ~ AssaHomRel + RobHomRel, data = houseAndCrime)
summary(multi4)

anova_unrest <- anova(multil2)
anova_rest <- anova(multi4)

#Test
statistic_test <- (((anova_rest$`Sum Sq`[3]-anova_unrest$`Sum Sq`[4])/2)
                   /((anova_unrest$`Sum Sq`[4])/anova_unrest$Df[4]))
print(statistic_test)
restrictions <- c(1, 1, 1, 0)
linearHypothesis(multi2,restrictions)
