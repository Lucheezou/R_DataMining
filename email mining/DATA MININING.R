#! - lines not needed


#INSTALL PACKAGES FIRST
library(tidyverse)
library(ggplot2)
library(GGally)
library(sampling)
library(rpart)
library(rpart.plot)
library(caret)
library(lattice)
library(FSelector)


#import dataset (tiktok playlist)
data <- read.csv("C:\\Users\\Samson\\Documents\\skool\\SE\\Data Mining R code and graphs\\email mining\\emails.csv")
data <- as_tibble(data)


#DATA PREPROCESSING

#remove duplicates and rows with missing values
data <- data %>% drop_na() %>% unique() #data now has 4645 objects instead of 5172


#remove unneccessary columns (Email.No.)
data <- data %>% select(-Email.No.) 

#making sure all numerical values are numerical not characters
data <- data %>% modify_if(is.character, numeric)

#converting binary class to labels
data$Prediction <- data$Prediction %>% replace(data$Prediction == 0, "NOT SPAM") 
data$Prediction <- data$Prediction %>% replace(data$Prediction == 1, "SPAM") 

#changing character to factor
data <- data %>%
  modify_if(is.character, factor)


#check summary of data
summary(data)
ggplot(data, aes(x=Prediction)) + geom_bar(stat="count")

#the data set contains more spam emails than non spam emails
#we will sample 1200 from each class to balance bias, since the lowest count has about 1500 vectors



#SAMPLE DATA
#stratified sampling of data using the filtered data
#sampling according to Prediction
set.seed(1000)
sample.data <- strata(data, stratanames = "Prediction", size = c(1200,1200), method = "srswor")
sample.data <- data[sample.data$ID_unit,]


#EDA


#check summary of data to see potential noise
summary(sample.data) #no out of the ordinary values to see

#Principal Component Analysis to reduce dimensionality
#!pc <- sample.data %>% select(-c(Prediction)) %>% as.matrix() %>% prcomp()

#checking which columns have the most variance
#!summary(pc)
#!plot(pc, type = "line")
#!str(pc)

#visualizing the PCA
#!data_projected <- as_tibble(pc$x) %>% add_column(Prediction = sample.data$Prediction)
#!ggplot(data_projected, aes(x = PC1, y = PC2, color = Prediction)) + 
  #!geom_point()


#see occurances of email classes in sample data
ggplot(sample.data, aes(x=Prediction)) + geom_bar(stat="count") #now we have 1200 objects from each Prediction category


#see relation of features to each other
#!ggpairs(sample.data, aes(color = Prediction), progress = FALSE) #we see it is hard to seperate songs on individual features


#CREATING A DECISION / REGRESSION TREE

#full tree
tree_full <- sample.data %>% rpart(Prediction ~., data = ., control = rpart.control(minsplit = 2, cp = 0))
rpart.plot(tree_full, extra = 2, roundint=FALSE,
           box.palette = list("Gy", "Gn", "Bu", "Bn", "Or", "Rd", "Pu")) # specify 7 colors

#default tree // use this for learning
tree_default <- sample.data %>% rpart(Prediction ~ ., data = .)
tree_default
rpart.plot(tree_default, extra = 2, roundint=FALSE)
           
#checking training error
predict(tree_default, sample.data) %>% head ()

pred <- predict(tree_default, sample.data, type="class")
head(pred)

confusion_table <- with(sample.data, table(Prediction, pred))
confusion_table

#confusion tables show many wrong predictions


#correctness
correct <- confusion_table %>% diag() %>% sum()
correct
#errors
error <- confusion_table %>% sum() - correct
error

#accuracy
accuracy <- correct / (correct + error)
accuracy #0.4 accuracy(pangit)


#setting accuracy as a function
accuracy <- function(truth, prediction) {
  tbl <- table(truth, prediction)
  sum(diag(tbl))/sum(tbl)
}

#accuracy of default tree 0.4 padin
accuracy(sample.data %>% pull(Prediction), pred)

#accuracy of full tree 0.995
accuracy(sample.data %>% pull(Prediction), predict(tree_full, sample.data, type="class"))

#confusiontable with caret
confusionMatrix(data = pred, reference = sample.data %>% pull(Prediction))



#MODEL EVAL WITH caret
set.seed(1000)

#splitting data for training and testing 80/20

inTrain <- createDataPartition(y = sample.data$Prediction, p = .8, list = FALSE)
data.train <- sample.data %>% slice(inTrain)
data.test <- sample.data %>% slice(-inTrain)

#learning the model
fit <- data.train %>%
  train(Prediction ~ .,
        data = . ,
        method = "rpart",
        control = rpart.control(minsplit = 2),
        trControl = trainControl(method = "cv", number = 10),
        tuneLength = 5)

fit

#check the tree of the best trained model
rpart.plot(fit$finalModel, extra = 2,
           box.palette = list("Gy", "Gn", "Bu", "Bn", "Or", "Rd", "Pu"))

#check variable importance
varImp(fit)

#variable importance without competing splits, used by caret
imp <- varImp(fit, compete = FALSE)
imp

#visualize importance
ggplot(imp)

#TESTING
#use best model to predict test data
pred <- predict(fit, newdata = data.test)
pred

#check confusion matrix
confusionMatrix(data = pred, ref = data.test$Prediction)

#comparing our tree classifier to a kNN classifier
train_index <- createFolds(data.train$Prediction, k = 10)


#building the tree classifier
rpartFit <- data.train %>% train(Prediction ~ .,
                                 data = .,
                                 method = "rpart",
                                 tuneLength = 10,
                                 trControl = trainControl(method = "cv", indexOut = train_index)
)

#check confusionmatrix of cart
pred1 <- predict(rpartFit, newdata = data.test)
pred1
confusionMatrix(data = pred1, reference = data.test$Prediction)

#building the knn classifier
knnFit <- data.train %>% train(Prediction ~  .,
                               data = .,
                               method = "knn",
                               preProcess = "scale",
                               tuneLength = 10,
                               trControl = trainControl(method = "cv", indexOut = train_index)
)

#check confusion matrix of knn
pred2 <- predict(knnFit, newdata = data.test)
pred2
confusionMatrix(data = pred2, reference = data.test$Prediction)


#compare the accuracies of both

resamps <- resamples(list(
  CART = rpartFit,
  kNearestNeighbors = knnFit
))

summary(resamps)

#visualize comparison
bwplot(resamps, layout = c(3, 1)) #kNN is better vs CART in this run

#FEATURE SELECTION

#using gain ratio function for continous values
weights <- data.train %>% gain.ratio(Prediction ~ ., data = .) %>%
  as_tibble(rownames = "feature") %>%
  arrange(desc(attr_importance))

weights

#view feature importance
ggplot(weights,
       aes(x = attr_importance, y = reorder(feature, attr_importance))) +
  geom_bar(stat = "identity") +
  xlab("Importance score") + ylab("Feature")

#get the 1000 best features
subset <- cutoff.k(weights %>% column_to_rownames("feature"), 1000)
subset

#using the best 1000 features to build model
f <- as.simple.formula(subset, "Prediction")
f

m <- data.train %>% rpart(f, data = .)
rpart.plot(m, extra = 2, roundint = FALSE)

#accuraccy test 
accuracy(sample.data %>% pull(Prediction), predict(m, sample.data, type="class")) # 0.3 accuracy

#conclusion: 
#model made with kNN has an accuracy of 50% vs model made with CART which has an accuraccy 30%
#both models trained with equal sampled data from the 4 classes with 400 objects each (1600 total)