2+2
rep(1, 100)

my_object <- seq(from = 0, to = 
                  50, by = 2)
plot(my_object)

getwd()

setwd("/Users/estefaniasosa/Documents/notes/edx-mit-ds-micromaster/data-analysis")
getwd()

install.packages("readxl")
library(readxl)

# .excel
read_excel("name")

# .csv
read.csv("name.cs")

# .dta 
install.packages("foreign")
library(foreign)

read.dta("name.dta")

# .sav

install.packages("haven")
library(haven)





