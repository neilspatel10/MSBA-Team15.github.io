rm(list=ls())
####################################################
### Functions
####################################################
installIfAbsentAndLoad <- function(neededVector) {
  for(thispackage in neededVector) {
    if( ! require(thispackage, character.only = T) )
    { install.packages(thispackage)}
    require(thispackage, character.only = T)
  }
}
##############################
### Load required packages ###
##############################

needed <- c('tree','randomForest', "gbm", "glmnet")
installIfAbsentAndLoad(needed)
time_start <- Sys.time()
##############################
######### Load Data ##########
##############################
set.seed(1)
data <- read.csv('data/train_enhanced.csv', header =T, stringsAsFactors = FALSE)
data <- data[,2:8]
data$feature_1  <- factor(data$feature_1)
data$feature_2 <- factor(data$feature_2)
data$feature_3 <- factor(data$feature_3)
n <- nrow(data)
train.indices <- sample(n, .8 * n)
data2 <- data
train_data2 <- data2[train.indices,]
test_data2 <- data2[-train.indices,]
data$first_active_month <- as.factor(data$first_active_month)
y <- data$target

data <- model.matrix(target~., data)[,-4]
data <- scale(data)
data <- data[,-1]

train_data <- as.data.frame(data[train.indices,])

train.y <- y[train.indices]
train_data <- data.frame(train_data, target = train.y)
test_data <- as.data.frame(data[-train.indices,])
test.y <- y[-train.indices]
test_data <- data.frame(test_data, target=test.y)
train_n <- nrow(train_data)

#######################################
####            Lasso              ####
#######################################
set.seed(5082)

grid = 10 ^ seq(10, -3, length=100)
mod.lasso <- glmnet(train.x, train.y, alpha=1, lambda=grid)
cv.out.class <- cv.glmnet(train.x, train.y, alpha=1, lambda=grid) # evaluate performance
bestlam <- cv.out.class$lambda.min # find best lambda
lasso.pred.class <- predict(mod.lasso,
                            s=bestlam,
                            newx=test.x,
                            type="class")

# RMSE
table(test.y, lasso.pred.class)
rmse.lasso <- sqrt(mean((test.y - lasso.pred.class)^2))
rmse.lasso
# rmse: 3.847178
lasso.coefficients <- predict(mod.lasso,
                              s=bestlam,
                              type="coefficients")[1:83,]
lasso.coefficients

#######################################
####            Ridge              ####
#######################################
set.seed(5082)

grid = 10 ^ seq(10, -3, length=100)
mod.ridge <- glmnet(train.x, train.y, alpha=0, lambda=grid)
cv.out.class <- cv.glmnet(train.x, train.y, alpha=0, lambda=grid) # evaluate performance
bestlam <- cv.out.class$lambda.min
ridge.pred.class <- predict(mod.ridge,
                            s=bestlam,
                            newx=test.x,
                            type="class")

# RMSE
table(test.y, ridge.pred.class)
rmse.ridge <- sqrt(mean((test.y - ridge.pred.class)^2))
rmse.ridge
#rmse: 3.847511
ridge.coefficients <- predict(mod.ridge,
                              s=bestlam,
                              type="coefficients")[1:83,]
ridge.coefficients

####################################################################
# Multiple Linear Regression
set.seed(1)
RMSE = function(m, o){
  sqrt(mean((m - o)^2))
}
test_data2 <- test_data2[test_data2$first_active_month != "2018-02",]
# Linear regression fit using features 1, 2, 3, Avg_purchase_amount, count_merchants, and first_active_month
lm.fit=lm(target~feature_1+feature_2+feature_3+Avg_purchase_amount+count_merchants+first_active_month,data=train_data2)

# Polynomial terms for the Avg_purchase amount and count_merchants
'''
poly_degrees <- c(1,2,3,4)
for (i in poly_degrees){
  for (j in poly_degrees){
      lm.poly=lm(target~feature_1+feature_2+feature_3+poly(Avg_purchase_amount,i)+poly(count_merchants,j)+first_active_month,data=train_data2)
      y_preds=predict(lm.poly, newdata=test_data2)
      print(paste("Average Purchase Degree: ",i))
      print(paste("Count Merchant Degree: ", j))
      print(RMSE(y_preds, test_data2$target))
  }
}'''
# lowest RMSE: 3.846742 with Avg_Purchase_amount degree 1 and count_merchants degree 4

# Interaction terms between the three features
lm.fit.12=lm(target~feature_1*feature_2+feature_3+Avg_purchase_amount+count_merchants+first_active_month,data=train_data2)
lm.fit.13=lm(target~feature_1*feature_3+feature_2+Avg_purchase_amount+count_merchants+first_active_month,data=train_data2)
lm.fit.23=lm(target~feature_2*feature_3+feature_1+Avg_purchase_amount+count_merchants+first_active_month,data=train_data2)

# Predict

y_preds <- predict(lm.fit, newdata=test_data2)
y_preds.12 <- predict(lm.fit.12, newdata=test_data2)
y_preds.13 <- predict(lm.fit.13, newdata=test_data2)
y_preds.23 <- predict(lm.fit.23, newdata=test_data2)

# Calculate RMSE

RMSE(y_preds, test_data2$target)
RMSE(y_preds.12, test_data2$target)
RMSE(y_preds.13, test_data2$target)
RMSE(y_preds.23, test_data2$target)

# interaction term between feature_2 and feature_3 did the best at 3.847453

# Combination of best polynomial and best interaction features
lm.fit.combo=lm(target~feature_2*feature_3+feature_1+Avg_purchase_amount+poly(count_merchants,4)+first_active_month,data=train_data2)
y_preds.combo=predict(lm.fit.combo, newdata=test_data2)
RMSE(y_preds.combo, test_data$target) # RMSE of 3.846591

# Best Model:
time_start <- Sys.time()
lm.fit.combo=lm(target~feature_2*feature_3+feature_1+Avg_purchase_amount+poly(count_merchants,4)+first_active_month,data=train_data2)
y_preds.combo=predict(lm.fit.combo, newdata=test_data2)
RMSE(y_preds.combo, test_data2$target) # RMSE of 3.846591
time_out <- Sys.time()
print(time_out - time_start)

############################
###### Random Forests ######
############################

# create a triple nested for loop to evaluate different parameters
num_trees <- c(5000, 10000, 15000)
shrinkage <- c(.1, .01, .001)
depth <- c(2, 4, 6)
matrix <- data.frame()
counter <- 0
'''
for ( n in num_trees){
  for (s in shrinkage){
    for(i in depth){
      boosted_rf <- gbm(target~., data = train_data, distribution="gaussian", n.trees=n, interaction.depth=i, shrinkage = s)
      y_preds <-predict(boosted_rf,newdata=test_data,n.trees=n)
      error <- mean((y_preds-test_data$target)^2)
      matrix <- rbind(matrix, c(error, n, s, i))
      counter <- counter + 1
      print(counter)
    }
  }
}
'''
# lines 66- 70 is to save the RMSE's of the models above in a csv file
#matrix[which.min(matrix[,1]),]
#RMSEs <- sqrt(matrix[,1])
#matrix$RMSEs <- RMSEs
#colnames(matrix) <- c("MeanSquaredError","Num_Trees","Learning_Rate",'Depth','RMSEs')
#write.csv(matrix, "data/Boosted_RF_errors.csv")

# bring in the file we wrote out to earlier
forest_data <- read.csv("data/Boosted_RF_errors.csv")
# then get the best model
best_forest <- forest_data[which.min(forest_data$RMSEs),]
# run it again
best_rf <- gbm(target~., data = train_data, distribution="gaussian",
               n.trees=best_forest$Num_Trees,
               interaction.depth=best_forest$Depth,
               shrinkage = best_forest$Learning_Rate)
y_preds <-predict(best_rf,newdata=test_data,n.trees=best_forest$Num_Trees)
error <- sqrt(mean((y_preds-test_data$target)^2))
error
summary(best_rf)
