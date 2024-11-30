##################################################
# ECON 418-518 Homework 2
# Kieran Douglas
# The University of Arizona
# kieran@arizona.edu 
# 24 November 2024
###################################################


#####################
# Preliminaries
#####################

# Clear environment, console, and plot pane
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation
options(scipen = 999)

# Load packages
pacman::p_load(data.table)

#####################
# Question 2
#####################
##############
# Part (i)
##############
install.packages("wooldridge")
library(wooldridge)
data("pntsprd")

# Create linear model for the situation at hand 
model1 <- lm(data = pntsprd, favwin~spread)
summary(model1)

##############
# Part (ii)
##############
#perform t test and obtain p value
t_test <- (coef(model1)[1] - 0.5) / sqrt(vcov(model1)[1,1])
p_value <- 2 * pt(abs(t_test), df = nrow(pntsprd) - 2, lower.tail = FALSE)

#find usual t and p values
cat("Usual t-statistic:", t_test, "\n")
cat("Usual p-value:", p_value, "\n")

# Test H0 using heteroskedasticity robust SE
robustse <- sqrt(vcovHC(model1, type = "HC1")[1,1])
robust_t <- (coef(model1)[1] - 0.5) / robust_se
robust_p <- 2 * pt(abs(robust_t), df = nrow(pntsprd) - 2, lower.tail = FALSE)

#find rovust t and p values
cat("Robust t-statistic:", robust_t, "\n")
cat("Robust p-value:", robust_p, "\n")

##############
# Part (iii)
##############
# pull coeficients to find p value for when spread is 10
beta0 <- coef(model1)[1]
beta1 <- coef(model1)[2]
pspread10 <- beta0 + beta1 * 10
cat("Estimated p when spread is equal to 10:", prob_spread_10, "\n")

##############
# Part (iv)
##############
prob_mod1 <- glm(data = pntsprd, favwin~spread, family = "binomial")
summary(prob_mod1)

# grabbing coefs to test the null that the intercept is 0
intercept_coef <- coef(prob_mod1)[1]
intercept_se <- sqrt(vcov(prob_mod1)[1,1])
z_statistic <- intercept_coef / intercept_se
p_value <- 2 * pnorm(-abs(z_statistic))

# print z and p
cat("Z-statistic for intercept:", z_statistic, "\n")
cat("P-value for intercept:", p_value, "\n")

##############
# Part (v)
##############
# when spread is 10
new_data <- data.frame(spread = 10)
probit_prob <- predict(prob_mod1, newdata = new_data, type = "response")

lpm_model <- lm(favwin ~ spread, data = pntsprd)

# Calculate probability for spread = 10 using LPM
lpm_prob <- predict(lpm_model, newdata = new_data)

# Print results
cat("Probit model probability when spread = 10:", probit_prob, "\n")
cat("LPM probability when spread = 10:", lpm_prob, "\n")


##############
# Part (vi)
##############
restricted_model <- glm(favwin ~ spread, family = binomial(link = "probit"), data = pntsprd)

# Estimate full model
full_model <- glm(favwin ~ spread + favhome + fav25 + und25, family = binomial(link = "probit"), data = pntsprd)

# Perform likelihood ratio test
lrtest_result <- lrtest(restricted_model, full_model)

# Print results
print(lrtest_result)

# Code



#####################
# Question 3
#####################
##############
# Part (i)
##############
data(loanapp)

# generate a linear probability model
modellin <- lm(data = loanapp, approve~white)
summary(modellin)
# output: 90.78% approval if white, 70.78% approval if not white 

# generate a probit model
modelpo <- glm(data = loanapp, approve~white, family = binomial(link = "probit"))
summary(modelpo)

# convert probit into probability of approval given that an applicant is white
# create new df where white is coded as 1 for yes
new_data <- data.frame(white = 0)  

# predict probability of approval given that the applicant is white (white = 1)
predicted_probability <- predict(modelpo, newdata = new_data, type = "response")
predicted_probability
# output p = 90.84% approval if white, 70.78% approval if not white  

##############
# Part (ii)
##############
# generate a probit model
modelpo2 <- glm(data = loanapp, approve~white+hrat+obrat+loanprc+unem+male+married+dep+sch+cosign+chist+pubrec+mortlat1+mortlat2+vr, family = binomial(link = "probit"))
summary(modelpo2)

# convert probit into probability of approval given that an applicant is white
# create new df where white is coded as 1 for yes
new_data <- data.frame(white = 1)  

# predict probability of approval given that the applicant is white (white = 1)
predicted_probability <- predict(modelpo2, newdata = new_data, type = "response")
predicted_probability
#hrat, obrat, loanprc, unem, male, married, dep, sch, cosign, chist, pubrec, mortlat1, mortlat2, and vr

##############
# Part (iii)
##############
modelpo3 <- glm(data = loanapp, approve~white+hrat+obrat+loanprc+unem+male+married+dep+sch+cosign+chist+pubrec+mortlat1+mortlat2+vr, family = binomial(link = "logit"))
summary(modelpo3)

##############
# Part (iv)
##############
install.packages("margins")
library(margins)

# for probit
probit_margins <- margins(modelpo2)
summary(probit_margins)

# for logit
logit_margins <- margins(modelpo3)
summary(logit_margins)



















