# Cousera Machine Learning Project code

# import training data, note that strings as factors is set to false to 
# make modeling algorithms function correctly
dat.train <- read.csv("pml-training.csv",TRUE,sep=",", stringsAsFactors = FALSE)
# import test data
dat.test <- read.csv("pml-testing.csv",TRUE,sep=",", stringsAsFactors = FALSE)
#na.strings="na")

# load packages to fit data
library(caret); library(kernlab); library("e1071"); library(randomForest)
library(dplyr); library(rpart.plot); library(rpart); library(rattle)
library(scales)
set.seed(32343)

# evaluate the data
str(dat.train)
# we can eliminate the X, user_name, kurtosis_yaw_belt, skewness_yaw_belt
# and all the timestamp columns because these add no value
drop_cols <- c('X','user_name','kurtosis_yaw_belt','skewness_yaw_belt','raw_timestamp_part_1','raw_timestamp_part_2','cvtd_timestamp')
dat.train <- dat.train[ , !(names(dat.train) %in% drop_cols)]

# run a summary to get a feel for the distribution of values within the variables
summary(dat.train)
# we see a good number of columns that have 19,216 NA values, adding little value
# and they can be eliminated. we'll use 95% NA rate to remove the columns
drop_cols <- which(colSums(dat.train=="" | is.na(dat.train))>0.95*dim(dat.train)[1]) 
str(drop_cols)
# we end up deleting 31 columns
dat.train <- dat.train[ ,-drop_cols]
# this leaves 122 columns
ncol(dat.train)
# next we eliminate redunant variables by checking for variables
# that are correlated with one another by at least 75% (source: https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/)
# get numeric columns
dat.train.nums <- select_if(dat.train,is.numeric)
# calculate correlation matrix
correlationMatrix <- cor(dat.train.nums)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
# remove highly correlated variables
dat.train <- dat.train[,-highlyCorrelated]
nrow(dat.train)

# partition data into train/test. Normally would use 70/30 split, but 
# going with 50/50 to save on processing
inTrain <- createDataPartition(y=dat.train$classe,
                               p=0.5, list=FALSE)
training <- dat.train[inTrain,]
testing <- dat.train[-inTrain,]

# next we to verify that the distribution of the classe variable is similar in training and testing data
table(training$classe)
table(testing$classe)
# decision tree example
mod_rpart <- train(classe ~ .,method="rpart",data=training)
print(mod_rpart$finalModel)

fancyRpartPlot(mod_rpart$finalModel)

# Predict and create confusion matrix
pred_cm <- predict(mod_rpart,newdata=testing)
cm_rpart <- confusionMatrix(as.factor(testing$classe),pred_cm)
cm_rpart
rpart_accuracy <- cm_rpart$overall['Accuracy']
# Try random forest
fitControl <- trainControl(method='cv', number = 3)
mod_rf <- train(classe~., data=training, method="rf", trControl=fitControl, verbose=FALSE)
# Plot
plot(mod_rf,main="RF Model Accuracy by number of predictors")

# Testing the model
pred_rf <- predict(mod_rf,newdata=testing)
cm_rf <- confusionMatrix(as.factor(testing$classe),pred_rf)
rf_accuracy <- cm_rf$overall['Accuracy']
# Display confusion matrix and model accuracy
cm_rf

# try general boosting model
mod_gbm <- train(classe~., data=training, method="gbm", trControl=fitControl, verbose=FALSE)
#  Plot 
plot(mod_gbm)

# Testing the model
pred_gbm <- predict(mod_gbm,newdata=testing)
cm_gbm <- confusionMatrix(as.factor(testing$classe),pred_gbm)

# Display confusion matrix and model accuracy
cm_gbm
gbm_accuracy <- cm_gbm$overall['Accuracy']

# Graph acccuracy of models to see which is more accurate
models <- c('Decision Tree','Random Forest','General Boosting')
accuracy <- c(rpart_accuracy,rf_accuracy,gbm_accuracy)
graph.data <- data.frame(models,accuracy)
p<-ggplot(data=graph.data, aes(x=models, y=accuracy)) +
        geom_bar(stat="identity", fill="steelblue")+
        geom_text(aes(label=percent(accuracy)), vjust=-0.3, size=3)+ scale_y_continuous(labels = scales::percent_format(accuracy = 1))+
        theme_minimal()
p

# We see that the general boosting model offers the highest accuracy.
# Next we predict the 20 sample cases by using the general boosting
# model. 
test_final <- predict(mod_rf,newdata=dat.test)
test_final
