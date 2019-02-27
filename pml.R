suppressMessages(library(caret))
training <- read.csv('pml-training.csv')
## remove columns with NA
varcol <- sapply(training, function(x) sum(is.na(x))) == 0
training <- training[,varcol]

## four sensors (1) belt (2) arm and (3) dumbbell (4) forearm
beltcolindx <- grep('belt',names(training))
armcolindx <- grep('_arm',names(training))
dumbbellcolindx <- grep('dumbbell',names(training))
forearmcolindx <- grep('forearm',names(training))
colindx <- c(beltcolindx,armcolindx,dumbbellcolindx,
             forearmcolindx,dim(training)[2])
## remove columns with near zero variance
nzv <- nearZeroVar(training)
colindx <- setdiff(colindx,nzv)
train <- training[,colindx]

## select build data and validation data
inValid <- createDataPartition(y=train$classe,p=0.7,list=FALSE)
builddata <- train[inValid,]
validdata <- train[-inValid,]

## prediction with decision tree
set.seed(1)
library(rpart)
system.time(fittree <- rpart(classe~.,data=builddata,method='class'))
## suppressMessages(library(rattle))
## fancyRpartPlot(fittree)
library(rpart.plot)
prp(fittree)
predtree <- predict(fittree,validdata,type='class')
cfmtree <- confusionMatrix(predtree, validdata$classe)
cfmtree
sprintf("Confusion matrix accuracy of decision tree is %.4f",cfmtree$overall[1])

## prediction with naive Bayes
library(MASS)
library(klaR)
ctrlnb <- trainControl(method='cv',number=3)
system.time(fitnb <-train(classe~.,data=builddata,method='nb',trControl=ctrlnb))
prednb <- predict(fitnb,validdata)
cfmnb <- confusionMatrix(prednb, validdata$classe)
cfmnb
sprintf("Confusion matrix accuracy of naive Bayes is %.4f",cfmnb$overall[1])

## prediction with random forest
ctrlrf <- trainControl(method='cv',number=3,verboseIter=FALSE)
system.time(fitrf <-train(classe~.,data=builddata,method='rf',trControl=ctrlrf))
## summary(fitrf$finalModel)
predrf <- predict(fitrf,validdata)
cfmrf <- confusionMatrix(predrf, validdata$classe)
cfmrf
sprintf("Confusion matrix accuracy of random forest is %.4f",cfmrf$overall[1])

## prediction with gradient boosting model
ctrlgbm <- trainControl(method='repeatedcv',number=3,repeats=1)
system.time(fitgbm <- train(classe~.,data=builddata,method='gbm',
                                    trControl=ctrlgbm,verbose=FALSE))
## summary(fitgbm$finalModel)
predgbm <- predict(fitgbm,validdata)
cfmgbm <- confusionMatrix(predgbm, validdata$classe)
cfmgbm
sprintf("Confusion matrix accuracy of boosted trees is %.4f",cfmgbm$overall[1])

## answer the quiz questions using the best model which is random forest
testing <- read.csv('pml-testing.csv')
testing <- testing[,varcol]
test <- testing[,colindx]
quizans <- predict(fitrf,test)
quizans
