##################################################
# ECON 418-518 Homework 3
# Kieran Douglas
# The University of Arizona
# kieran@arizona.edu 
# 28 November 2024
###################################################


#####################
# Preliminaries
#################

# Clear environment, console, and plot pane
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation
options(scipen = 999)

# Load packages
pacman::p_load(data.table)

# Set sead
set.seed(418518)

# install packages and load data
install.packages("tidyverse")
install.packages("caret")
library(caret)
library(tidyverse)
hw3 <- read_csv("~/Documents/GitHub/bios/ECON 418 518 Homework 3 Data.csv")
#####################
# Problem 1
#####################
#################
# Question (i)
#################
filtered <- hw3 %>% 
  select(age, 
         workclass, 
         education, 
         `marital-status`, 
         relationship, 
         race, 
         gender, 
         `hours-per-week`, 
         `native-country`, 
         income)
# selected only variables specified for new df

#################
# Question (ii)
#################
clean <- filtered %>%
  mutate(income_binary = if_else(income == ">50K" | income == ">50K.", 1, 0),
         race_bi = if_else(race == "White" | race == "White", 1, 0),
         gender_bi =  if_else(gender == "Male", 1, 0),
         workclass_bi = if_else(workclass == "Private", 1, 0),
         nativeco_bi = if_else(`native-country` == "United-States", 1, 0),
         marr_bi = if_else(`marital-status` == "Married-civ-spouse", 1, 0),
         education_bi = if_else(education %in% c("Bachelors", "Masters", "Doctorate"), 1, 0),
         age_sq = age^2,
         age_std = scale(age),
         age_squared_std = scale(age^2),
         hpw = scale(`hours-per-week`))
# mutates each thing to binary, added age squared, and made scale conversions for interpretations in terms of sd rather than raw


#################
# Question (iii)
#################
sum(clean$income_binary==1)
# [1] 11687
sum(clean$workclass_bi==1)
# [1] 33906
sum(clean$marr_bi==1)
# [1] 22379
sum(clean$gender_bi==0)
# [1] 16192
column_missing <- colSums(clean == "?", na.rm = TRUE)
print(column_missing)
# 2799+857 = [1] 3656
clean <- clean %>% 
  mutate(income = as.factor(income))
# income converted to factor 

#################
# Question (iv)
#################
last <- floor(nrow(clean) * 0.70)
# find last training set
train <- clean[1:last, ]
# training
test <- clean[(last + 1):nrow(clean), ]
#testing

#################
# Question (iv)
#################
# Load necessary libraries
library(caret)
library(glmnet)

# Set seed for reproducibility
set.seed(123)

# Assuming 'clean' is your dataset and 'income' is the outcome variable
# Ensure 'income' is a factor for classification
clean$income <- as.factor(clean$income)

# Prepare predictors and outcome variable
# Use model.matrix without removing the intercept for now
x <- model.matrix(income ~ ., data = clean)
# Remove the intercept column by specifying the intercept term in the formula
x <- x[, colnames(x) != "(Intercept)"]
y <- clean$income

# Define lambda grid: 50 evenly spaced values from 1e5 to 1e-2 on a log scale
lambda_grid <- exp(seq(log(1e5), log(1e-2), length.out = 50))

# Set up 10-fold cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Train Lasso regression model using caret's train() function
lasso_grid <- expand.grid(alpha = 1, lambda = lambda_grid)

lasso_model <- train(
  x = x,
  y = y,
  method = "glmnet",
  family = "binomial",
  trControl = train_control,
  tuneGrid = lasso_grid,
  standardize = TRUE
)

# Extract the best lambda and highest classification accuracy
best_lambda <- lasso_model$bestTune$lambda
best_accuracy <- max(lasso_model$results$Accuracy)

cat("Best lambda for Lasso:", best_lambda, "\n")
cat("Highest classification accuracy for Lasso:", best_accuracy, "\n")

# Extract coefficients at the best lambda
lasso_coefficients <- coef(lasso_model$finalModel, s = best_lambda)

# Convert coefficients to a named numeric vector
lasso_coefs <- as.numeric(lasso_coefficients)
names(lasso_coefs) <- rownames(lasso_coefficients)

# Remove intercept
lasso_coefs <- lasso_coefs[names(lasso_coefs) != "(Intercept)"]

# Identify variables with coefficients approximately zero
nonzero_coef_vars <- names(lasso_coefs)[abs(lasso_coefs) >= 1e-4]
zero_coef_vars <- names(lasso_coefs)[abs(lasso_coefs) < 1e-4]

cat("Variables with coefficients approximately zero:\n")
print(zero_coef_vars)

cat("Variables with non-zero coefficients:\n")
print(nonzero_coef_vars)

# Ensure variable names match the column names in x
variables_to_keep <- intersect(nonzero_coef_vars, colnames(x))

cat("Variables to keep (after matching with x):\n")
print(variables_to_keep)

# Subset predictors to non-zero coefficient variables, ensuring matrix structure is maintained
x_subset <- x[, variables_to_keep, drop = FALSE]

# Verify that x_subset has column names
if (is.null(colnames(x_subset))) {
  stop("x_subset does not have column names. Please check the variable names.")
}

# Train Lasso regression model on subset variables
lasso_model_subset <- train(
  x = x_subset,
  y = y,
  method = "glmnet",
  family = "binomial",
  trControl = train_control,
  tuneGrid = lasso_grid,
  standardize = TRUE
)

# Train Ridge regression model on subset variables
ridge_grid <- expand.grid(alpha = 0, lambda = lambda_grid)

ridge_model_subset <- train(
  x = x_subset,
  y = y,
  method = "glmnet",
  family = "binomial",
  trControl = train_control,
  tuneGrid = ridge_grid,
  standardize = TRUE
)

# Get the highest classification accuracy for both models
best_lasso_accuracy_subset <- max(lasso_model_subset$results$Accuracy)
best_ridge_accuracy_subset <- max(ridge_model_subset$results$Accuracy)

cat("Highest classification accuracy for Lasso with subset variables:", best_lasso_accuracy_subset, "\n")
cat("Highest classification accuracy for Ridge with subset variables:", best_ridge_accuracy_subset, "\n")

# Determine which model has better accuracy
if (best_lasso_accuracy_subset > best_ridge_accuracy_subset) {
  cat("Lasso regression model has better classification accuracy.\n")
} else if (best_ridge_accuracy_subset > best_lasso_accuracy_subset) {
  cat("Ridge regression model has better classification accuracy.\n")
} else {
  cat("Both models have the same classification accuracy.\n")
}

#################
# Question (v)
#################
library(caret)
library(glmnet)

# Assuming your data is in a dataframe called 'data'
# with 'income' as the outcome variable and all other variables as predictors

# Create the training control object for 10-fold cross-validation
ctrl <- trainControl(method = "cv", number = 10)

# Create a sequence of 50 lambda values
lambda_seq <- 10^seq(5, -2, length.out = 50)

# Train the lasso model
lasso_model <- train(
  income ~ .,
  data = filtered,
  method = "glmnet",
  trControl = ctrl,
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_seq)
)

# Print the results
print(lasso_model)

best_lambda <- lasso_model$bestTune$lambda
best_accuracy <- max(lasso_model$results$Accuracy)

cat("Best lambda:", best_lambda, "\n")
cat("Best accuracy:", best_accuracy, "\n")

lasso_coef <- coef(lasso_model$finalModel, s = lasso_model$bestTune$lambda)
zero_coef <- lasso_coef[abs(lasso_coef) < 1e-5]
print(zero_coef)

# Get non-zero coefficient variables
non_zero_vars <- rownames(lasso_coef)[abs(lasso_coef) >= 1e-5]

# Create new formula with selected variables
new_formula <- as.formula(paste("income ~", paste(non_zero_vars[-1], collapse = " + ")))

# Train lasso model with selected variables
lasso_selected <- train(
  new_formula,
  data = data,
  method = "glmnet",
  trControl = ctrl,
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_seq)
)

# Train ridge model with selected variables
ridge_selected <- train(
  new_formula,
  data = data,
  method = "glmnet",
  trControl = ctrl,
  tuneGrid = expand.grid(alpha = 0, lambda = lambda_seq)
)

# Compare accuracies
lasso_accuracy <- max(lasso_selected$results$Accuracy)
ridge_accuracy <- max(ridge_selected$results$Accuracy)

cat("Lasso accuracy:", lasso_accuracy, "\n")
cat("Ridge accuracy:", ridge_accuracy, "\n")
cat("Best model:", ifelse(lasso_accuracy > ridge_accuracy, "Lasso", "Ridge"))

#################
# Question (vi)
#################
# Load necessary libraries
library(randomForest)
library(caret)

# Set seed for reproducibility
set.seed(123)

# Assuming your data frame is named 'data' and 'income' is the outcome variable
# Ensure that 'income' is a factor for classification
clean$income <- as.factor(clean$income)

# Split the data into predictors and outcome
predictors <- clean[, !(names(clean) %in% "income")]
outcome <- clean$income

# Define training control with 5-fold cross-validation
train_control <- trainControl(method = "cv", number = 5)

# Define the grid of hyperparameters
# We'll create a combination of ntree and mtry
# However, 'caret' tunes 'mtry' by default; 'ntree' can be set using the 'tuneGrid' or 'tuneLength'

# Since 'caret' does not directly tune 'ntree', we'll need to train separate models for each 'ntree'

# Define the different 'mtry' values
mtry_values <- c(2, 5, 9)

# Define the different 'ntree' values
ntree_values <- c(100, 200, 300)

# Initialize a list to store models
models <- list()

# Loop over each 'ntree' value and train models with different 'mtry'
for (ntree in ntree_values) {
  # Define the tuning grid for 'mtry'
  tune_grid <- expand.grid(.mtry = mtry_values)
  
  # Train the model using 'caret'
  model <- train(
    income ~ ., 
    data = clean,
    method = "rf",
    metric = "Accuracy",
    trControl = train_control,
    tuneGrid = tune_grid,
    ntree = ntree
  )
  
  # Store the model with a name indicating its parameters
  model_name <- paste0("RF_ntree_", ntree)
  models[[model_name]] <- model
}

# Print model results
for (name in names(models)) {
  cat("Model:", name, "\n")
  print(models[[name]])
  cat("\n")
}



