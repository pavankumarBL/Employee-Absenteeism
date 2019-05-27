rm(list=ls())

setwd("C:\\Users\\pavankumar.bl\\Documents\\datascience\\Edwisor\\Project")

x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

lapply(x,require,character.only=TRUE)

rm(x)
library(xlsx)
#Training_data=read.xlsx("Absenteeism_at_work_Project.xls",sheetIndex=1,header=TRUE)
Data_input=read.xlsx("Absenteeism_at_work_Project.xls",sheetIndex=1,header=TRUE)

str(Data_input)
Data_input$Reason.for.absence[Data_input$Reason.for.absence %in% 0] = 25
Data_input$Month.of.absence[Data_input$Month.of.absence %in% 0] = NA
# Training_data$ID=as.factor(Training_data$ID)
# Training_data$Reason.for.absence=as.factor(Training_data$Reason.for.absence)
# Training_data$Day.of.the.week=as.factor(Training_data$Day.of.the.week)
# Training_data$Seasons=as.factor(Training_data$Seasons)
# Training_data$Education=as.factor(Training_data$Education)
# Training_data$Son=as.factor(Training_data$Son)
# Training_data$Social.drinker=as.factor(Training_data$Social.drinker)
# Training_data$Social.smoker =as.factor(Training_data$Social.smoker )
# Training_data$Pet=as.factor(Training_data$Pet)
# Training_data$Hit.target=as.factor(Training_data$Hit.target)

#For Loop instead of so many lines of code
for (i in c(1,2,4,5,11,12:17)){
  Data_input[,i]=as.factor(as.character(Data_input[,i]))
}
View(Data_input)
str(Data_input)

#===================Missing value analysis==================#
Missing_val=data.frame(apply(Data_input,2,function(x){sum(is.na(x))}))
Missing_val$Columns=row.names(Missing_val)
row.names(Missing_val)=NULL
names(Missing_val)[1]= "Missing_values"
#finding missing value percentage for all variable in data set
Missing_val$Missing_percentage=(Missing_val$Missing_values/nrow(Data_input))*100
#descending order
Missing_val=Missing_val[order(-Missing_val$Missing_percentage),]
#plotting the graph
ggplot(data = Missing_val[1:15,], aes(x=reorder(Columns, -Missing_percentage),y = Missing_percentage))+
geom_bar(stat = "identity",fill = "grey")+xlab("Parameter")+
ggtitle("Missing data percentage (Train)") + theme_bw()
#imputing the data with Mean median
#Value of Body mass index at Data_input[5,20]= 30
#mean=26.67938
#median=25
Data_input[["Body.mass.index"]][5]
Data_input[["Body.mass.index"]][5]=NA
Data_input$Body.mass.index[is.na(Data_input$Body.mass.index)]=mean(Data_input$Body.mass.index,na.rm = T)
Data_input$Body.mass.index[is.na(Data_input$Body.mass.index)]=median(Data_input$Body.mass.index,na.rm = T)


#Lets check KNN Result
#Value of Body mass index at Data_input[5,20]= 30
#KNN Value =30
Data_input=knnImputation(Data_input,k=5)
sum(is.na(Data_input))




################################OutLIER Analysis#################################
numeric_index=sapply(Data_input,is.numeric)
numeric_data=Data_input[,numeric_index]
cnames=colnames(numeric_data)
for(i in 1:ncol(numeric_data)) {
  assign(paste0("box",i), ggplot(data = Data_input, aes_string(y = numeric_data[,i])) +
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour = "red", fill = "grey", outlier.size = 1) +
           labs(y = colnames(numeric_data[i])) +
           ggtitle(paste("Boxplot: ",colnames(numeric_data[i]))))
}
#Arrange the plots in grids
gridExtra::grid.arrange(box1,box2,box3,box4,ncol=4)
gridExtra::grid.arrange(box5,box6,box7,box8,ncol=4)
gridExtra::grid.arrange(box9,box10,ncol=2)


numeric_columns = colnames(numeric_data)
df=Data_input
#Replace all outlier data with NA
for(i in numeric_columns){
  val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
  print(paste(i,length(val)))
  df[,i][df[,i] %in% val] = NA
}

sum(is.na(df))
#Compute the NA values using KNN imputation
df = knnImputation(df, k = 5)
View(df)
##########################################Feature selection###############
#correlation plot
corrgram(df[,numeric_index],order = F,upper.panel = panel.pie,text.panel = panel.txt,main="Correlatio plot")

#Variable Reduction
df = subset.data.frame(df, select = -c(Body.mass.index))
#Make a copy of Copy of Data
new_data = df
write.xlsx(new_data, "new_data.xlsx", row.names = F)
#########################Feature scaling########################
#check Normality
qqnorm(df$Absenteeism.time.in.hours)
hist(df$Absenteeism.time.in.hours)

#Remove dependent variable
numeric_index = sapply(df,is.numeric)
numeric_data = df[,numeric_index]
numeric_columns = names(numeric_data)
numeric_columns = numeric_columns[-9]

#Normalization of continuous variables
for(i in numeric_columns){
  print(i)
  df[,i] = (df[,i] - min(df[,i]))/
    (max(df[,i]) - min(df[,i]))
}
#Get the data with only factor columns
factor_data = df[,!numeric_index]
#Get the names of factor variables
factor_columns = names(factor_data)

#Create dummy variables of factor variables
df = dummy.data.frame(df, factor_columns)

rmExcept(keepers = c("df"))
#######################Decision Tree###############
#Divide the data into test and train 
set.seed(1)
train_index = sample(1:nrow(df), 0.8*nrow(df))
train = df[train_index,]
test = df[-train_index,]

library("rpart")
#rpart regression 
fit= rpart(formula = Absenteeism.time.in.hours ~ .,data = train, method = "anova")

#Plot the tree
library("rpart.plot")
rpart.plot(fit)

#Perdict for test cases
predictions = predict(fit, test[,-115])

#Create data frame for actual and predicted values
df_pred = data.frame("actual"=test[,115], "dt_pred"=predictions)
head(df_pred)

#Calcuate MAE, RMSE, R-sqaured for testing data 
print(postResample(pred = predictions, obs = test[,115]))
#RMSE 2.3372625
#Rsquared 0.4201232
#MAE 1.7358038

#Plot a graph for actual vs predicted values
plot(test$Absenteeism.time.in.hours,type="l",lty=2,col="green")+
lines(predictions,col="blue")

#######################Random forest##############################
##Train the model using training data
rf_model = randomForest(Absenteeism.time.in.hours~., data = train, ntree = 500)
#Predict the test cases
rf_predictions = predict(rf_model, test[,-115])  
#Create dataframe for actual and predicted values
df_pred = cbind(df_pred,rf_predictions)
head(df_pred)
#Calcuate MAE, RMSE, R-sqaured for testing data 
print(postResample(pred = rf_predictions, obs = test[,115]))
#RMSE 2.1688392
#Rsquared 0.4978645
#MAE 1.5694553

#Plot a graph for actual vs predicted values
plot(test$Absenteeism.time.in.hours,type="l",lty=2,col="green")+
lines(rf_predictions,col="red")
########################################LINEAR REGRESSION########################################


##Train the model using training data
lr_model = lm(formula = Absenteeism.time.in.hours~., data = train)
#Get the summary of the model
summary(lr_model)
#Predict the test cases
lr_predictions = predict(lr_model, test[,-115])
#Create dataframe for actual and predicted values
df_pred = cbind(df_pred,lr_predictions)
head(df_pred)
#Calcuate MAE, RMSE, R-sqaured for testing data 
print(postResample(pred = lr_predictions, obs = test[,115]))
#RMSE 2.3899807
#Rsquared 0.4212922 
#MAE 1.7272814
#Plot a graph for actual vs predicted values
plot(test$Absenteeism.time.in.hours,type="l",lty=2,col="green")+
lines(lr_predictions,col="blue")

#--------------------------------------------XGBoost-------------------------------------------#

set.seed(123)

#Develop Model on training data
fit_XGB = gbm(Absenteeism.time.in.hours~., data = train, n.trees = 500, interaction.depth = 2)

#Lets predict for training data
pred_XGB_train = predict(fit_XGB, train[,names(test) != "Absenteeism.time.in.hours"], n.trees = 500)

#Lets predict for testing data
pred_XGB_test = predict(fit_XGB,test[,names(test) != "Absenteeism.time.in.hours"], n.trees = 500)

# For training data 
print(postResample(pred = pred_XGB_train, obs = train[,107]))

# For testing data 
print(postResample(pred = pred_XGB_test, obs = test[,107]))

########################################DIMENSION REDUCTION USING PCA########################################

#Principal component analysis
prin_comp = prcomp(train)

#Compute standard deviation of each principal component
pr_stdev = prin_comp$sdev

#Compute variance
pr_var = pr_stdev^2
#Proportion of variance explained
prop_var = pr_var/sum(pr_var)

#Cumulative scree plot
plot(cumsum(prop_var), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")
#Add a training set with principal components
train.data = data.frame(Absenteeism.time.in.hours = train$Absenteeism.time.in.hours, prin_comp$x)

# From the above plot selecting 45 components since it explains almost 95+ % data variance
train.data =train.data[,1:45]

#Transform test data into PCA
test.data = predict(prin_comp, newdata = test)
test.data = as.data.frame(test.data)

#Select the first 45 components
test.data=test.data[,1:45]
########################################DECISION TREE########################################

#Build decsion tree using rpart
dt_model = rpart(Absenteeism.time.in.hours ~., data = train.data, method = "anova")

#Predict the test cases
dt_predictions = predict(dt_model,test.data)

#Create data frame for actual and predicted values
df_pred = data.frame("actual"=test[,115], "dt_pred"=dt_predictions)
head(df_pred)

#Calcuate MAE, RMSE, R-sqaured for testing data 
print(postResample(pred = dt_predictions, obs = test$Absenteeism.time.in.hours))
#RMSE: 0.4700759
#MAE: 0.3048803
#R squared: 0.9765771
#Plot a graph for actual vs predicted values
plot(test$Absenteeism.time.in.hours,type="l",lty=2,col="red")+
lines(dt_predictions,col="blue")
########################################RANDOM FOREST########################################


#Train the model using training data
rf_model = randomForest(Absenteeism.time.in.hours~., data = train.data, ntrees = 500)

#Predict the test cases
rf_predictions = predict(rf_model,test.data)

#Create dataframe for actual and predicted values
df_pred = cbind(df_pred,rf_predictions)
head(df_pred)

#Calcuate MAE, RMSE, R-sqaured for testing data 
print(postResample(pred = rf_predictions, obs = test$Absenteeism.time.in.hours))
#RMSE: 0.5182287
#MAE: 0.2897083
#R squared: 0.9752510
#Plot a graph for actual vs predicted values
plot(test$Absenteeism.time.in.hours,type="l",lty=2,col="green")+
lines(rf_predictions,col="blue")
########################################LINEAR REGRESSION########################################


#Train the model using training data
lr_model = lm(Absenteeism.time.in.hours ~ ., data = train.data)

#Get the summary of the model
summary(lr_model)

#Predict the test cases
lr_predictions = predict(lr_model,test.data)

#Create dataframe for actual and predicted values
df_pred = cbind(df_pred,lr_predictions)
head(df_pred)

#Calcuate MAE, RMSE, R-sqaured for testing data 
print(postResample(pred = lr_predictions, obs =test$Absenteeism.time.in.hours))
#RMSE: 0.002580320
#MAE: 0.001898268
#R squared: 0.999999353
#Plot a graph for actual vs predicted values
plot(test$Absenteeism.time.in.hours,type="l",lty=2,col="green")+
lines(lr_predictions,col="blue")
