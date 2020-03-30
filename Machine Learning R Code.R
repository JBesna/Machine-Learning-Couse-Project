# Machine Learning

# Week 1

# SPAM detection
library(kernlab)
data(spam)
head(spam)

plot(density(spam$your[spam$type=="nonspam"]),
     col="blue",main="",xlab="Frequency of 'your'")
lines(density(spam$your[spam$type=="spam"]),col="red")

plot(density(spam$your[spam$type=="nonspam"]),
     col="blue",main="",xlab="Frequency of 'your'")
lines(density(spam$your[spam$type=="spam"]),col="red")
abline(v=0.5,col="black")

prediction <- ifelse(spam$your > 0.5,"spam","nonspam")
table(prediction,spam$type)/length(spam$type)

# Overfitting - More SPAM prediction

library(kernlab); data(spam); set.seed(333)
smallSpam <- spam[sample(dim(spam)[1],size=10),]
spamLabel <- (smallSpam$type=="spam")*1 + 1
plot(smallSpam$capitalAve,col=spamLabel)

# rule 1 is overfit and more complex
rule1 <- function(x){
        prediction <- rep(NA,length(x))
        prediction[x > 2.7] <- "spam"
        prediction[x < 2.40] <- "nonspam"
        prediction[(x >= 2.40 & x <= 2.45)] <- "spam"
        prediction[(x > 2.45 & x <= 2.70)] <- "nonspam"
        return(prediction)
}
table(rule1(smallSpam$capitalAve),smallSpam$type)

# rule2 is more general, and later proves more accurate
rule2 <- function(x){
        prediction <- rep(NA,length(x))
        prediction[x > 2.8] <- "spam"
        prediction[x <= 2.8] <- "nonspam"
        return(prediction)
}
table(rule2(smallSpam$capitalAve),smallSpam$type)

# Apply sample rules to population
table(rule1(spam$capitalAve),spam$type)
table(rule2(spam$capitalAve),spam$type)

# Calculate prediction accuracy - rule2 is more accurate and simpler
mean(rule1(spam$capitalAve)==spam$type)
mean(rule2(spam$capitalAve)==spam$type)

# Quiz 1
# 1) Training and Test Data Sets
# 2) Overfitting noise
# 3) 60/40
# 4) Predictive value of a positive
# 5) 9%

# Week 2 - The Caret Package

library(caret); library(kernlab); data(spam)
inTrain <- createDataPartition(y=spam$type,
                               p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
dim(training)

# fit model using caret function train()
library("e1071")
set.seed(32343)
modelFit <- train(type ~.,data=training, method="glm")

modelFit <- train(type ~.,data=training, method="glm")
modelFit$finalModel
modelFit

# make predictions on test data set
predictions <- predict(modelFit,newdata=testing)
predictions

# display confusion matrix on predictions
confusionMatrix(predictions,testing$type)

# data splicing
library(caret); library(kernlab); data(spam)
inTrain <- createDataPartition(y=spam$type,
                               p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
dim(training)

# K-folds, values are not replaced (no duplicates)
# creates multiple sets of roughly the same size and has different
# permutaitons of train/test in each
set.seed(32323)
folds <- createFolds(y=spam$type,k=10,
                     list=TRUE,returnTrain=TRUE)
sapply(folds,length)

set.seed(32323)
folds <- createFolds(y=spam$type,k=10,
                     list=TRUE,returnTrain=FALSE)
sapply(folds,length)

# resampling aka BOOTSTRAPPING, values are replaced (could be duplicates)
set.seed(32323)
folds <- createResample(y=spam$type,times=10,
                        list=TRUE)
sapply(folds,length)

# time slices
set.seed(32323)
tme <- 1:1000
folds <- createTimeSlices(y=tme,initialWindow=20,
                          horizon=10)
names(folds)

folds$train[[1]]
folds$test[[1]]

# Training Options

library(caret); library(kernlab); data(spam)
inTrain <- createDataPartition(y=spam$type,
                               p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
modelFit <- train(type ~.,data=training, method="glm")

# use set.seed(#) to generate repeatable random sample
set.seed(1235)
modelFit2 <- train(type ~.,data=training, method="glm")
modelFit2

set.seed(1235)
modelFit3 <- train(type ~.,data=training, method="glm")
modelFit3

# Plotting Predictors

library(ISLR); library(ggplot2); library(caret);
data(Wage)
summary(Wage)
inTrain <- createDataPartition(y=Wage$wage,
                               p=0.7, list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
dim(training); dim(testing)

# plot features
featurePlot(x=training[,c("age","education","jobclass")],
            y = training$wage,
            plot="pairs")
# qq plots
qplot(age,wage,data=training)
qplot(age,wage,colour=jobclass,data=training)
# add regression smoothers
qq <- qplot(age,wage,colour=education,data=training)
qq +  geom_smooth(method='lm',formula=y~x)

# partition data in thirds
library(Hmisc)
library(gridExtra)
cutWage <- cut2(training$wage,g=3)
table(cutWage)
# make boxplots with split data 
p1 <- qplot(cutWage,age, data=training,fill=cutWage,
            geom=c("boxplot"))
p1
# box plot with data overlay
p2 <- qplot(cutWage,age, data=training,fill=cutWage,
            geom=c("boxplot","jitter"))
grid.arrange(p1,p2,ncol=2)
# tables
t1 <- table(cutWage,training$jobclass)
t1
prop.table(t1,1)

# density plots
qplot(wage,colour=education,data=training,geom="density")

# PreProcessing
library(caret); library(kernlab); data(spam)
inTrain <- createDataPartition(y=spam$type,
                               p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
hist(training$capitalAve,main="",xlab="ave. capital run length")
mean(training$capitalAve)
sd(training$capitalAve)

# standarize data to tighten distribution and bring std dev = 1
trainCapAve <- training$capitalAve
trainCapAveS <- (trainCapAve  - mean(trainCapAve))/sd(trainCapAve) 
mean(trainCapAveS)
sd(trainCapAveS)

# standardize test data separately
testCapAve <- testing$capitalAve
testCapAveS <- (testCapAve  - mean(trainCapAve))/sd(trainCapAve) 
mean(testCapAveS)
sd(testCapAveS)

# standardize using preProcess function
preObj <- preProcess(training[,-58],method=c("center","scale"))
trainCapAveS <- predict(preObj,training[,-58])$capitalAve
mean(trainCapAveS)
sd(trainCapAveS)
p2 <- qplot(cutWage,age, data=training,fill=cutWage,
            geom=c("boxplot","jitter"))
grid.arrange(p1,p2,ncol=2)

testCapAve <- testing$capitalAve
testCapAveS <- (testCapAve  - mean(trainCapAve))/sd(trainCapAve) 
mean(testCapAveS)
testCapAveS <- predict(preObj,testing[,-58])$capitalAve
mean(testCapAveS)
sd(testCapAveS)

set.seed(32343)
modelFit <- train(type ~.,data=training,
                  preProcess=c("center","scale"),method="glm")
modelFit

# Box-Cox transforms to evalute normality
preObj <- preProcess(training[,-58],method=c("BoxCox"))
trainCapAveS <- predict(preObj,training[,-58])$capitalAve
par(mfrow=c(1,2)); hist(trainCapAveS); qqnorm(trainCapAveS)

# Imputing missing data
set.seed(13343)

# Make some values NA
training$capAve <- training$capitalAve
selectNA <- rbinom(dim(training)[1],size=1,prob=0.05)==1
training$capAve[selectNA] <- NA

# Impute and standardize
preObj <- preProcess(training[,-58],method="knnImpute")
capAve <- predict(preObj,training[,-58])$capAve

# Standardize true values
capAveTruth <- training$capitalAve
capAveTruth <- (capAveTruth-mean(capAveTruth))/sd(capAveTruth)

# Standardizing - Imputing data
quantile(capAve - capAveTruth)
quantile((capAve - capAveTruth)[selectNA])
quantile((capAve - capAveTruth)[!selectNA])

# Covariate Creation
library(kernlab);data(spam)
spam$capitalAveSq <- spam$capitalAve^2

library(ISLR); library(caret); data(Wage);
inTrain <- createDataPartition(y=Wage$wage,
                               p=0.7, list=FALSE)
training <- Wage[inTrain,]; testing <- Wage[-inTrain,]

table(training$jobclass)
# create dummy variables
dummies <- dummyVars(wage ~ jobclass,data=training)
head(predict(dummies,newdata=training))
# remove zero covariates - predictors with little variation
nsv <- nearZeroVar(training,saveMetrics=TRUE)
nsv

# spline basis
library(splines)
bsBasis <- bs(training$age,df=3) 
bsBasis

# fitting curves with splines
lm1 <- lm(wage ~ bsBasis,data=training)
plot(training$age,training$wage,pch=19,cex=0.5)
points(training$age,predict(lm1,newdata=training),col="red",pch=19,cex=0.5)

# spline on test set
predict(bsBasis,age=testing$age)

# PCA - Principal Component Analysis
# correlated predictors
library(caret); library(kernlab); data(spam)
inTrain <- createDataPartition(y=spam$type,
                               p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]

M <- abs(cor(training[,-58]))
diag(M) <- 0
which(M > 0.8,arr.ind=T)

names(spam)[c(34,32)]
plot(spam[,34],spam[,32])

# rotate the plot
X <- 0.71*training$num415 + 0.71*training$num857
Y <- 0.71*training$num415 - 0.71*training$num857
plot(X,Y) # more variance on horizontal than vertical

# use prcomp function in caret
smallSpam <- spam[,c(34,32)]
prComp <- prcomp(smallSpam)
plot(prComp$x[,1],prComp$x[,2])

prComp$rotation

# use graphics
typeColor <- ((spam$type=="spam")*1 + 1)
prComp <- prcomp(log10(spam[,-58]+1))
plot(prComp$x[,1],prComp$x[,2],col=typeColor,xlab="PC1",ylab="PC2")

# graph with caret
preProc <- preProcess(log10(spam[,-58]+1),method="pca",pcaComp=2)
spamPC <- predict(preProc,log10(spam[,-58]+1))
plot(spamPC[,1],spamPC[,2],col=typeColor)

# preprocessing with PCA
preProc <- preProcess(log10(training[,-58]+1),method="pca",pcaComp=2)
trainPC <- predict(preProc,log10(training[,-58]+1))
modelFit <- train(training$type ~ .,method="glm",data=trainPC)

testPC <- predict(preProc,log10(testing[,-58]+1))
confusionMatrix(testing$type,predict(modelFit,testPC))

# use caret
modelFit <- train(training$type ~ .,method="glm",preProcess="pca",data=training)
confusionMatrix(testing$type,predict(modelFit,testing))

# Predicting with Regression

# Old faithful eruptions
library(caret);data(faithful); set.seed(333)
inTrain <- createDataPartition(y=faithful$waiting,
                               p=0.5, list=FALSE)
trainFaith <- faithful[inTrain,]; testFaith <- faithful[-inTrain,]
head(trainFaith)

plot(trainFaith$waiting,trainFaith$eruptions,pch=19,col="blue",xlab="Waiting",ylab="Duration")
lm1 <- lm(eruptions ~ waiting,data=trainFaith)
summary(lm1)

plot(trainFaith$waiting,trainFaith$eruptions,pch=19,col="blue",xlab="Waiting",ylab="Duration")
lines(trainFaith$waiting,lm1$fitted,lwd=3)

# predict new value
coef(lm1)[1] + coef(lm1)[2]*80
newdata <- data.frame(waiting=80)
predict(lm1,newdata)

# plot predictions - training and test
par(mfrow=c(1,2))
plot(trainFaith$waiting,trainFaith$eruptions,pch=19,col="blue",xlab="Waiting",ylab="Duration")
lines(trainFaith$waiting,predict(lm1),lwd=3)
plot(testFaith$waiting,testFaith$eruptions,pch=19,col="blue",xlab="Waiting",ylab="Duration")
lines(testFaith$waiting,predict(lm1,newdata=testFaith),lwd=3)

# calculate RMSE
sqrt(sum((lm1$fitted-trainFaith$eruptions)^2))
sqrt(sum((predict(lm1,newdata=testFaith)-testFaith$eruptions)^2))

# prediction intervals
pred1 <- predict(lm1,newdata=testFaith,interval="prediction")
ord <- order(testFaith$waiting)
plot(testFaith$waiting,testFaith$eruptions,pch=19,col="blue")
matlines(testFaith$waiting[ord],pred1[ord,],type="l",,col=c(1,2,2),lty = c(1,1,1), lwd=3)

# now using caret package
modFit <- train(eruptions ~ waiting,data=trainFaith,method="lm")
summary(modFit$finalModel)

# predicting wages example
library(ISLR); library(ggplot2); library(caret);
data(Wage); Wage <- subset(Wage,select=-c(logwage))
summary(Wage)

# create training and test
inTrain <- createDataPartition(y=Wage$wage,
                               p=0.7, list=FALSE)
training <- Wage[inTrain,]; testing <- Wage[-inTrain,]
dim(training); dim(testing)

# feature plot
featurePlot(x=training[,c("age","education","jobclass")],
            y = training$wage,
            plot="pairs")

# plot age vs wage
qplot(age,wage,data=training)

# with job classification coloring
qplot(age,wage,colour=jobclass,data=training)

qplot(age,wage,colour=education,data=training)

# fit model
modFit<- train(wage ~ age + jobclass + education,
               method = "lm",data=training)
finMod <- modFit$finalModel
print(modFit)

# diagnostics
plot(finMod,1,pch=19,cex=0.5,col="#00000010")

# color by variables not in model
qplot(finMod$fitted,finMod$residuals,colour=race,data=training)

# plot by index
plot(finMod$residuals,pch=19)

# predictions vs actual
pred <- predict(modFit, testing)
qplot(wage,pred,colour=year,data=testing)
# using all covariances
modFitAll<- train(wage ~ .,data=training,method="lm")
pred <- predict(modFitAll, testing)
qplot(wage,pred,data=testing)

# Quiz - Week 2
# 1 
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
# answer:

adData = data.frame(diagnosis,predictors)
trainIndex = createDataPartition(diagnosis, p = 0.50,list=FALSE)
training = adData[trainIndex,]
testing = adData[-trainIndex,]
# 2
library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]
plot(inTrain$CompressiveStrength,pch=19)
# There is a non-random pattern in the plot of the outcome versus index that does not appear to be perfectly explained by any predictor suggesting a variable may be missing.

#3 
library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]
# histrogram
hist(training$Superplasticizer)
table(training$Superplasticizer)
# There are values of zero so when you take the log() transform those values will be -Inf.

#4 
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
# of variables: 9

# 5
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

#non-PCA 0.65, PCA 0.72

# Week 3 - Trees

data(iris); library(ggplot2)
names(iris)
table(iris$Species)

# split data
inTrain <- createDataPartition(y=iris$Species,
                               p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training); dim(testing)
training <- na.omit(training)
# plot training
qplot(Petal.Width,Sepal.Width,colour=Species,data=training)

# fit model
library(caret)
modFit <- train(Species ~ .,method="rpart",data=training)
print(modFit$finalModel)

# plot tree - basic dendrogram
plot(modFit$finalModel, uniform=TRUE, 
     main="Classification Tree")
text(modFit$finalModel, use.n=TRUE, all=TRUE, cex=.8)

# better formatted plot 
library(rattle)
fancyRpartPlot(modFit$finalModel)

# predicting values
predict(modFit,newdata=testing)
testing$preds <- predict(modFit,newdata=testing)
head(testing)

# compare actual vs. prediction
validation <- testing$preds == testing$Species
table(validation)

# Bagging / bootstrapping - useful for non-linear functions
# takes an average of smoothed samples
library(ElemStatLearn); data(ozone,package="ElemStatLearn")
ozone <- ozone[order(ozone$ozone),]
head(ozone)

# generate smoothed loess curves
ll <- matrix(NA,nrow=10,ncol=155)
for(i in 1:10){
        ss <- sample(1:dim(ozone)[1],replace=T)
        ozone0 <- ozone[ss,]; ozone0 <- ozone0[order(ozone0$ozone),]
        loess0 <- loess(temperature ~ ozone,data=ozone0,span=0.2)
        ll[i,] <- predict(loess0,newdata=data.frame(ozone=1:155))
}

# graph smoothed loess curves
plot(ozone$ozone,ozone$temperature,pch=19,cex=0.5)
for(i in 1:10){lines(1:155,ll[i,],col="grey",lwd=2)}
lines(1:155,apply(ll,2,mean),col="red",lwd=2)

# bagging in caret
library(party)
predictors = data.frame(ozone=ozone$ozone)
temperature = ozone$temperature
treebag <- bag(predictors, temperature, B = 10,
               bagControl = bagControl(fit = ctreeBag$fit,
                                       predict = ctreeBag$pred,
                                       aggregate = ctreeBag$aggregate))
# custom bagging
plot(ozone$ozone,temperature,col='lightgrey',pch=19)
points(ozone$ozone,predict(treebag$fits[[1]]$fit,predictors),pch=19,col="red")
points(ozone$ozone,predict(treebag,predictors),pch=19,col="blue")
ctreeBag$fit

# Random Forests - libraries: caret, randomForest
library(caret)

data(iris); library(ggplot2)
inTrain <- createDataPartition(y=iris$Species,
                               p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]

modFit <- train(Species~ .,data=training,method="rf",prox=TRUE)
modFit

# look at specific tree (#2)
getTree(modFit$finalModel,k=2)

# class centers to see the center of the prediction
library(randomForest)
irisP <- classCenter(training[,c(3,4)], training$Species, modFit$finalModel$prox)
irisP <- as.data.frame(irisP); irisP$Species <- rownames(irisP)
p <- qplot(Petal.Width, Petal.Length, col=Species,data=training)
p + geom_point(aes(x=Petal.Width,y=Petal.Length,col=Species),size=5,shape=4,data=irisP)

# predict new values from random forests and 
# generate confusion matrix
pred <- predict(modFit,testing); testing$predRight <- pred==testing$Species
table(pred,testing$Species)

# predicting new falues
qplot(Petal.Width,Petal.Length,colour=predRight,data=testing,main="newdata Predictions")

# Boosting
library(ISLR); data(Wage); library(ggplot2); library(caret);
Wage <- subset(Wage,select=-c(logwage))
inTrain <- createDataPartition(y=Wage$wage,
                               p=0.7, list=FALSE)
training <- Wage[inTrain,]; testing <- Wage[-inTrain,]

# fit boosted model
modFit <- train(wage ~ ., method="gbm",data=training,verbose=FALSE)
print(modFit)

qplot(predict(modFit,testing),wage,data=testing)

# Model Based Prediction
data(iris); library(ggplot2)
names(iris)
inTrain <- createDataPartition(y=iris$Species,
                               p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training); dim(testing)

# build 2 types of predictions
modlda = train(Species ~ .,data=training,method="lda")
modnb = train(Species ~ ., data=training,method="nb")
plda = predict(modlda,testing); pnb = predict(modnb,testing)
table(plda,pnb)

# compare results of 2 model types
equalPredictions = (plda==pnb)
qplot(Petal.Width,Sepal.Width,colour=equalPredictions,data=testing)

# Week 3 Quiz
# 1)
library(AppliedPredictiveModeling)
data(segmentationOriginal)
library(caret)
dat <- segmentationOriginal
inTrain <- createDataPartition(y=dat$Case,
                               p=0.7, list=FALSE)
training <- dat[inTrain,]
testing <- dat[-inTrain,]
modFit <- train(Class ~ .,method="rpart",data=training)
print(modFit$finalModel)
library(rattle)
fancyRpartPlot(modFit$finalModel)
TotalIntench2 <- 23000
FiberWidthCh1 <- 10
PerimStatusCh1 <- 2
Cell <- 1
test <- data.frame(TotalIntench2, FiberWidthCh1, PerimStatusCh1)
predict(modFit, newdata=test)
# d not possible to predict for missing TotalIntench2

# 2)
# X - The bias is smaller and the variance is bigger. Under leave one out cross validation K is equal to one.
# The bias is larger
# 3)

library(pgmm)
data(olive)
olive = olive[,-1]
modFit <- train(Area ~ .,method="rpart",data=olive)
newdata = as.data.frame(t(colMeans(olive)))
predict(modFit, newdata=newdata)
# 2.783

# 4)
library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]
set.seed(13234)
# fit logistic model
modelFit <- glm(chd ~ age + alcohol + obesity + tobacco + typea + ldl,data=trainSA, family = "binomial")
missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}
preds <- ifelse(modelFit$fitted.values > 0.5, 1, 0)
confusionMatrix(trainSA$chd, preds)
trainSA$preds <- preds
trainSA$match <- ifelse(trainSA$preds == trainSA$chd,1,0)
trainaccuracy <- sum(trainSA$match)/nrow(trainSA)
1 - trainaccuracy
# training set = 0.272

# 5)
library(ElemStatLearn)
data(vowel.train)
data(vowel.test)
set.seed(33833)

# X order of variables is 10, 7, ...
# order is 2, 1, 5

# Week 4 - Regularized Regressors and 

library(ElemStatLearn); data(prostate)
str(prostate)

covnames <- names(prostate[-(9:10)])
y <- prostate$lpsa
x <- prostate[,covnames]

form <- as.formula(paste("lpsa~", paste(covnames, collapse="+"), sep=""))
summary(lm(form, data=prostate[prostate$train,]))

set.seed(1)
train.ind <- sample(nrow(prostate), ceiling(nrow(prostate))/2)
y.test <- prostate$lpsa[-train.ind]
x.test <- x[-train.ind,]

y <- prostate$lpsa[train.ind]
x <- x[train.ind,]

p <- length(covnames)
rss <- list()
for (i in 1:p) {
        cat(i)
        Index <- combn(p,i)
        
        rss[[i]] <- apply(Index, 2, function(is) {
                form <- as.formula(paste("y~", paste(covnames[is], collapse="+"), sep=""))
                isfit <- lm(form, data=x)
                yhat <- predict(isfit)
                train.rss <- sum((y - yhat)^2)
                
                yhat <- predict(isfit, newdata=x.test)
                test.rss <- sum((y.test - yhat)^2)
                c(train.rss, test.rss)
        })
}

png("Plots/selection-plots-01.png", height=432, width=432, pointsize=12)
plot(1:p, 1:p, type="n", ylim=range(unlist(rss)), xlim=c(0,p), xlab="number of predictors", ylab="residual sum of squares", main="Prostate cancer data")
for (i in 1:p) {
        points(rep(i-0.15, ncol(rss[[i]])), rss[[i]][1, ], col="blue")
        points(rep(i+0.15, ncol(rss[[i]])), rss[[i]][2, ], col="red")
}
minrss <- sapply(rss, function(x) min(x[1,]))
lines((1:p)-0.15, minrss, col="blue", lwd=1.7)
minrss <- sapply(rss, function(x) min(x[2,]))
lines((1:p)+0.15, minrss, col="red", lwd=1.7)
legend("topright", c("Train", "Test"), col=c("blue", "red"), pch=1)
dev.off()

##
# ridge regression on prostate dataset
library(MASS)
lambdas <- seq(0,50,len=10)
M <- length(lambdas)
train.rss <- rep(0,M)
test.rss <- rep(0,M)
betas <- matrix(0,ncol(x),M)
for(i in 1:M){
        Formula <-as.formula(paste("y~",paste(covnames,collapse="+"),sep=""))
        fit1 <- lm.ridge(Formula,data=x,lambda=lambdas[i])
        betas[,i] <- fit1$coef
        
        scaledX <- sweep(as.matrix(x),2,fit1$xm)
        scaledX <- sweep(scaledX,2,fit1$scale,"/")
        yhat <- scaledX%*%fit1$coef+fit1$ym
        train.rss[i] <- sum((y - yhat)^2)
        
        scaledX <- sweep(as.matrix(x.test),2,fit1$xm)
        scaledX <- sweep(scaledX,2,fit1$scale,"/")
        yhat <- scaledX%*%fit1$coef+fit1$ym
        test.rss[i] <- sum((y.test - yhat)^2)
}

png(file="Plots/selection-plots-02.png", width=432, height=432, pointsize=12) 
plot(lambdas,test.rss,type="l",col="red",lwd=2,ylab="RSS",ylim=range(train.rss,test.rss))
lines(lambdas,train.rss,col="blue",lwd=2,lty=2)
best.lambda <- lambdas[which.min(test.rss)]
abline(v=best.lambda+1/9)
legend(30,30,c("Train","Test"),col=c("blue","red"),lty=c(2,1))
dev.off()


png(file="Plots/selection-plots-03.png", width=432, height=432, pointsize=8) 
plot(lambdas,betas[1,],ylim=range(betas),type="n",ylab="Coefficients")
for(i in 1:ncol(x))
        lines(lambdas,betas[i,],type="b",lty=i,pch=as.character(i))
abline(h=0)
legend("topright",covnames,pch=as.character(1:8))
dev.off()


#######
# lasso
library(lars)
lasso.fit <- lars(as.matrix(x), y, type="lasso", trace=TRUE)

png(file="Plots/selection-plots-04.png", width=432, height=432, pointsize=8) 
plot(lasso.fit, breaks=FALSE)
legend("topleft", covnames, pch=8, lty=1:length(covnames), col=1:length(covnames))
dev.off()

# this plots the cross validation curve
png(file="Plots/selection-plots-05.png", width=432, height=432, pointsize=12) 
lasso.cv <- cv.lars(as.matrix(x), y, K=10, type="lasso", trace=TRUE)
dev.off()

# Combining Predictors
library(ISLR); data(Wage); library(ggplot2); library(caret);
Wage <- subset(Wage,select=-c(logwage))

# Create a building data set and validation set
inBuild <- createDataPartition(y=Wage$wage,
                               p=0.7, list=FALSE)
validation <- Wage[-inBuild,]; buildData <- Wage[inBuild,]

inTrain <- createDataPartition(y=buildData$wage,
                               p=0.7, list=FALSE)
training <- buildData[inTrain,]; testing <- buildData[-inTrain,]

dim(training)

dim(testing)

dim(validation)

# fit GLM and Random Forest models
mod1 <- train(wage ~.,method="glm",data=training)
mod2 <- train(wage ~.,method="rf",
              data=training, 
              trControl = trainControl(method="cv"),number=3)

pred1 <- predict(mod1,testing); pred2 <- predict(mod2,testing)
qplot(pred1,pred2,colour=wage,data=testing)

# fit a model the combines the predictors
predDF <- data.frame(pred1,pred2,wage=testing$wage)
combModFit <- train(wage ~.,method="gam",data=predDF)
combPred <- predict(combModFit,predDF)

# testing errors
sqrt(sum((pred1-testing$wage)^2))
sqrt(sum((pred2-testing$wage)^2))
sqrt(sum((combPred-testing$wage)^2)) # combined model has lowest errors

# predict on validation set
pred1V <- predict(mod1,validation); pred2V <- predict(mod2,validation)
predVDF <- data.frame(pred1=pred1V,pred2=pred2V)
combPredV <- predict(combModFit,predVDF)

# evaluate errors
sqrt(sum((pred1V-validation$wage)^2))
sqrt(sum((pred2V-validation$wage)^2))
sqrt(sum((combPredV-validation$wage)^2)) # combined model has lowest error

# Forecasting - Time Series
# Google stock price
library(quantmod)
from.dat <- as.Date("01/01/08", format="%m/%d/%y")
to.dat <- as.Date("12/31/13", format="%m/%d/%y")
getSymbols("GOOG", src="yahoo", from = from.dat, to = to.dat)
head(GOOG)

# Summarize monthly and store as time series
mGoog <- to.monthly(GOOG)
googOpen <- Op(mGoog)
ts1 <- ts(googOpen,frequency=12)

# Decompose movement into trend and seasonality
plot(decompose(ts1),xlab="Years+1")

# Train and test
ts1Train <- window(ts1,start=1,end=5)
ts1Test <- window(ts1,start=5,end=(7-0.01))
ts1Train

# simple moving average
library(forecast)
plot(ts1Train)
lines(ma(ts1Train,order=3),col="red")

# exponential smoothing
ets1 <- ets(ts1Train,model="MMM")
fcast <- forecast(ets1)
plot(fcast); lines(ts1Test,col="red")

# test accuracy
accuracy(fcast,ts1Test)

# Unsupervised Prediction

# start with k-means
data(iris); library(ggplot2)
inTrain <- createDataPartition(y=iris$Species,
                               p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training); dim(testing)

# plot clusters
kMeans1 <- kmeans(subset(training,select=-c(Species)),centers=3)
training$clusters <- as.factor(kMeans1$cluster)
qplot(Petal.Width,Petal.Length,colour=clusters,data=training)

# compare to real labels
table(kMeans1$cluster,training$Species)

# build predictors
modFit <- train(clusters ~.,data=subset(training,select=-c(Species)),method="rpart")
table(predict(modFit,training),training$Species)

# apply on test
testClusterPred <- predict(modFit,testing) 
table(testClusterPred ,testing$Species)

# Quiz Week 4

# 1)
library(ElemStatLearn)
library(gbm)
data(vowel.train)
data(vowel.test)
# set y variable to factor
vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)
set.seed(33833)

# fit random forest model
modRF <- train(y ~ .,data=vowel.train,method="rf",prox=TRUE)
# fit boosted model
modGLM <- train(y ~ .,data=vowel.train,method="gbm",prox=TRUE)
# prediction
testRF <- predict(modRF,vowel.test) 
vowel.test$preds <- testRF
vowel.test$match <- ifelse(vowel.test$preds == vowel.test$y,1,0)
trainaccuracy <- sum(vowel.test$match)/nrow(vowel.test)
#d .6082, .512, .512
# .61, .51, .63

# 2)
library(caret)

library(gbm)

set.seed(3433)

library(AppliedPredictiveModeling)

data(AlzheimerDisease)

adData = data.frame(diagnosis,predictors)

inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]

training = adData[ inTrain,]

testing = adData[-inTrain,]
set.seed(62433)
# fit random forest model
modRF <- train(diagnosis ~ .,data=training,method="rf",prox=TRUE)
# fit boosted model
modGLM <- train(diagnosis ~ .,data=training,method="gbm",prox=TRUE)
# fit linear discriminate analysis model
modLDA <- train(diagnosis ~ .,data=training,method="lda",prox=TRUE)
# prediction of random forest
testRF <- predict(modRF,testing) 
testing$preds <- testRF
testing$match <- ifelse(testing$preds == testing$diagnosis,1,0)
trainaccuracy <- sum(testing$match)/nrow(testing)
trainaccuracy
# prediction of lda
testLDA <- predict(modLDA,testing) 
testing$preds <- testLDA
testing$match <- ifelse(testing$preds == testing$diagnosis,1,0)
trainaccuracy <- sum(testing$match)/nrow(testing)
trainaccuracy

# stacked accuracy is 0.8 and the same as boosting
# better than all 3
# 0.76 better than lda
# 3)
set.seed(3523)

library(AppliedPredictiveModeling)

data(concrete)

inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]

training = concrete[ inTrain,]

testing = concrete[-inTrain,]
set.seed(233)
lasso.cv <- cv.lars(as.matrix(training), y, K=10, type="lasso", trace=TRUE)
# age
# coarse aggregate
# cement
# 4)
library(lubridate) # For year() function below
library(forecast)
dat = read.csv("~/Desktop/gaData.csv")

training = dat[year(dat$date) < 2012,]

testing = dat[(year(dat$date)) > 2011,]

tstrain = ts(training$visitsTumblr)
modBATS <- bats(tstrain)
ci <- confint(modBATS)
head(tstrain)
# 93%
# 96%
# 5)
set.seed(3523)

library(AppliedPredictiveModeling)

data(concrete)

inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]

training = concrete[ inTrain,]

testing = concrete[-inTrain,]
set.seed(325)
library(e1071)
library(ModelMetrics)
model_svm <- svm(CompressiveStrength ~. , training)
predictions <- predict(model_svm,newdata=testing)
rmse(testing$CompressiveStrength, predictions)
# 6.71