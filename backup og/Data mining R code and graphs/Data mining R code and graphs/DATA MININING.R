
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
data <- read.csv("C:\\Users\\Samson\\Documents\\skool\\SE\\tiktok.csv")
data <- as_tibble(data)


#DATA PREPROCESSING

#remove duplicates and rows with missing values
data <- data %>% drop_na() %>% unique() #data now has 4645 objects instead of 6000


#remove unneccessary columns (track_id,artist_id,album_id,duration,playlist_id,playlist_name,genre,mode,release_date)
data <- data %>% select(-c(track_id ,artist_id,album_id,duration,playlist_id,playlist_name,genre,mode,release_date,track_name,artist_name)) #now only has 12 columns instead of 23

#making sure all numerical values are numerical not characters
data <- data %>% modify_if(is.character, numeric)

#generating 4 class names grouped according to popularity score
data$popularity[data$popularity >= 80]<-"Very Popular"
data$popularity[data$popularity >= 50 & data$popularity < 80]<-"Popular"
data$popularity[data$popularity >= 20 & data$popularity < 50]<-"Fairly Popular"
data$popularity[data$popularity >= 0 & data$popularity < 20]<-"Slightly Popular"
data$popularity[data$popularity == 9] <- "Slightly Popular" #edge case lang to di nasasama ang 9 na value sa pagconvert kaya linagay ko to


#changing character to factor
data <- data %>%
  modify_if(is.character, factor)


#check summary of data
summary(data)
ggplot(data, aes(x=popularity)) + geom_bar(stat="count")

#the data set contains more popular songs compared to fairly, slightly and very popular
#we will sample 200 from each class to balance bias, since the lowest count has about 500 (498) vectors





#SAMPLE DATA
#stratified sampling of data using the filtered data
#sampling according to popularity
set.seed(1000)
sample.data <- strata(data, stratanames = "popularity", size = c(400,400,400,400), method = "srswor")
sample.data <- data[sample.data$ID_unit,]


#EDA

#see occurances of popular songs in sample data
ggplot(sample.data, aes(x=popularity)) + geom_bar(stat="count") #now we have 200 songs from each popularity category


#see relation of popular songs based on other features
ggpairs(sample.data, aes(color = popularity), progress = FALSE) #we see it is hard to seperate songs on individual features


#check summary of data to see potential noise
summary(sample.data) #no out of the ordinary values to see

#Principal Component Analysis
pc <- sample.data %>% select(-c(popularity)) %>% as.matrix() %>% prcomp()

#checking which columns have the most variance
summary(pc)
plot(pc, type = "line")
str(pc)

#visualizing the PCA
data_projected <- as_tibble(pc$x) %>% add_column(popularity = sample.data$popularity)
ggplot(data_projected, aes(x = PC1, y = PC2, color = popularity)) + 
  geom_point()

#PC1 (danceability) has the most variance per vector



#CREATING A DECISION / REGRESSION TREE

#full tree
tree_full <- sample.data %>% rpart(popularity ~., data = ., control = rpart.control(minsplit = 2, cp = 0))
rpart.plot(tree_full, extra = 2, roundint=FALSE,
           box.palette = list("Gy", "Gn", "Bu", "Bn", "Or", "Rd", "Pu")) # specify 7 colors

#default tree // use this for learning
tree_default <- sample.data %>% rpart(popularity ~ ., data = .)
tree_default
rpart.plot(tree_default, extra = 2, roundint=FALSE)
           
#checking training error
predict(tree_default, sample.data) %>% head ()

pred <- predict(tree_default, sample.data, type="class")
head(pred)

confusion_table <- with(sample.data, table(popularity, pred))
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
accuracy(sample.data %>% pull(popularity), pred)

#accuracy of full tree 0.995
accuracy(sample.data %>% pull(popularity), predict(tree_full, sample.data, type="class"))

#confusiontable with caret
confusionMatrix(data = pred, reference = sample.data %>% pull(popularity))

#pass a prediction to default tree

#make new vector with custom properties
my_song <- tibble(danceability = 0.9, energy=0.5, key=12, loudness = -4.2,
                    speechiness = 0.2, acousticness = 0.3, instrumentalness = 1.77e-05, liveness = 0.5,
                    valence = 0.866, tempo = 120, duration_mins = 2.0)

predict(tree_default , my_song, type = "class") #slightly popular result

#MODEL EVAL WITH caret
set.seed(1000)

#splitting data for training and testing 80/20

inTrain <- createDataPartition(y = sample.data$popularity, p = .8, list = FALSE)
data.train <- sample.data %>% slice(inTrain)
data.test <- sample.data %>% slice(-inTrain)

#learning the model
fit <- data.train %>%
  train(popularity ~ .,
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
confusionMatrix(data = pred, ref = data.test$popularity)

#comparing our tree classifier to a kNN classifier
train_index <- createFolds(data.train$popularity, k = 10)


#building the tree classifier
rpartFit <- data.train %>% train(popularity ~ .,
                                data = .,
                                method = "rpart",
                                tuneLength = 10,
                                trControl = trainControl(method = "cv", indexOut = train_index)
)

#check confusionmatrix of cart
pred1 <- predict(rpartFit, newdata = data.test)
pred1
confusionMatrix(data = pred1, reference = data.test$popularity)

#building the knn classifier
knnFit <- data.train %>% train(popularity ~  .,
                              data = .,
                              method = "knn",
                              preProcess = "scale",
                              tuneLength = 10,
                              trControl = trainControl(method = "cv", indexOut = train_index)
)

#check confusion matrix of knn
pred2 <- predict(knnFit, newdata = data.test)
pred2
confusionMatrix(data = pred2, reference = data.test$popularity)

#compare the accuracies of both

resamps <- resamples(list(
  CART = rpartFit,
  kNearestNeighbors = knnFit
))

summary(resamps)

#visualize comparison
bwplot(resamps, layout = c(3, 1)) #kNN is better vs CART in this run

#FEATURE SELECTION

#using gain ratio function for continuous values
weights <- data.train %>% gain.ratio(popularity ~ ., data = .) %>%
  as_tibble(rownames = "feature") %>%
  arrange(desc(attr_importance))

weights

#view feature importance
ggplot(weights,
       aes(x = attr_importance, y = reorder(feature, attr_importance))) +
  geom_bar(stat = "identity") +
  xlab("Importance score") + ylab("Feature")

#get the 5 best features
subset <- cutoff.k(weights %>% column_to_rownames("feature"), 5)
subset

#using the best 5 features to build model
f <- as.simple.formula(subset, "popularity")
f

m <- data.train %>% rpart(f, data = .)
rpart.plot(m, extra = 2, roundint = FALSE)

#accuraccy test 
accuracy(sample.data %>% pull(popularity), predict(m, sample.data, type="class")) # 0.3 accuracy

#conclusion: 
#model made with kNN has an accuracy of 50% vs model made with CART which has an accuraccy 30%
#both models trained with equal sampled data from the 4 classes with 400 objects each (1600 total)