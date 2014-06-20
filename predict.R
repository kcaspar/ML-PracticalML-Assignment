library("caret")
data=read.csv("pmltrain.csv", header=TRUE)
testdata=read.csv("pmltest.csv", header=TRUE)

##separate the data from the outcome
outcome<-subset(data,select=c("classe"))
predictors=within(data,rm("classe"))
testpredictors=within(testdata,rm("problem_id"))

##drop predictors that are not numeric
numericlist=sapply(predictors, is.numeric)
predictors=predictors[,numericlist]
testpredictors=testpredictors[,numericlist]

##remove predictors that contain almost only NA
nacount=colSums(is.na(predictors))
keep=(nacount<nrow*0.9)
predictors=predictors[,keep]
testpredictors=testpredictors[,keep]

##keep complete cases only - PCA does it too, but left it for other training algo
complete=complete.cases(predictors)
predictors=predictors[complete,]
outcome=outcome[complete,]

##pca to reduce data and speed up training time
print("preprocessing data - running PCA ...")
pcaprep=preProcess(predictors,method="pca",thresh=0.9)
data=predict(pcaprep,predictors)
testdata=predict(pcaprep,testpredictors)
print(pcaprep)

##split training data into test and cross validation set 60-40
##important to split the outcome list too
print("preprocessing data - split into train and CV")
set.seed(1)
split=createDataPartition(outcome,p=0.6,list=FALSE)
tdata=data[split,]
toutcome=outcome[split]
cvdata=data[-split,]
cvoutcome=outcome[-split]

##train the model
print("training the model ...")
y=train(toutcome~.,data=tdata,method="rf")
y=y$finalModel

##predict the training, crossvalidation and test set
print("predicting outcomes ...")
predtrain=predict(y,newdata=tdata)
predcv=predict(y,newdata=cvdata)
predtest=predict(y,newdata=testdata)

missclass=function(values,prediction){
pos=(values==prediction)
count=sum(pos)
total=length(values)
result=count/total
}

print("accuracy training set prediction:")
err=missclass(toutcome,predtrain)
print(err)

print("accuracy crossvalidation set prediction:")
err=missclass(cvoutcome,predcv)
print(err)

print("predicted outcomes for test set:")
print(predtest)
