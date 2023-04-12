
#INSTALL PACKAGES FIRST
library(randomForest)
library(tidyverse)
library(ggplot2)
library(GGally)
library(sampling)
library(rpart)
library(rpart.plot)
library(caret)
library(lattice)
library(FSelector)
library(RWeka)
library(keras)
library(scales)
library(e1071)


options(digits=3)

#import dataset (Zoo from mlbench)
data(Zoo, package="mlbench")
Zoo <- as_tibble(Zoo)
Zoo


#Setting Aside Test Data (80%/20%)
inTrain <- createDataPartition(y = Zoo$type, p = .8, list = FALSE)
Zoo_train <- Zoo %>% slice(inTrain)
Zoo_test <- Zoo %>% slice(-inTrain)

#sampling scheme
train_index <- createFolds(Zoo_train$type, k = 10)


#fitting different kinds of models
#Conditional Inference Tree
ctreeFit <- Zoo_train %>% train(type ~ .,
                                method = "ctree",
                                data = .,
                                tuneLength = 5,
                                trControl = trainControl(method = "cv", indexOut = train_index))
ctreeFit

#plot the CIT tree
plot(ctreeFit$finalModel)

#C 4.5 Decision Tree
C45Fit <- Zoo_train %>% train(type ~ .,
                              method = "J48",
                              data = .,
                              tuneLength = 5,
                              trControl = trainControl(method = "cv", indexOut = train_index))
C45Fit

#View C 4.5 model
C45Fit$finalModel

#KNN model
knnFit <- Zoo_train %>% train(type ~ .,
                              method = "knn",
                              data = .,
                              preProcess = "scale",
                              tuneLength = 5,
                              tuneGrid=data.frame(k = 1:10),
                              trControl = trainControl(method = "cv", indexOut = train_index))
knnFit

#view knn model
knnFit$finalModel


#PART (Rule Based Classifier) model
rulesFit <- Zoo_train %>% train(type ~ .,
                                method = "PART",
                                data = .,
                                tuneLength = 5,
                                trControl = trainControl(method = "cv", indexOut = train_index))
rulesFit

#view PART model
rulesFit$finalModel

#SVM model
svmFit <- Zoo_train %>% train(type ~.,
                              method = "svmLinear",
                              data = .,
                              tuneLength = 5,
                              trControl = trainControl(method = "cv", indexOut = train_index))
svmFit

#view SVM model
svmFit$finalModel

#Random Forest Model
randomForestFit <- Zoo_train %>% train(type ~ .,
                                       method = "rf",
                                       data = .,
                                       tuneLength = 5,
                                       trControl = trainControl(method = "cv", indexOut = train_index))
randomForestFit

#view Random Forest Model
randomForestFit$finalModel

#Gradient Boosted Decision Trees
xgboostFit <- Zoo_train %>% train(type ~ .,
                                  method = "xgbTree",
                                  data = .,
                                  tuneLength = 5,
                                  trControl = trainControl(method = "cv", indexOut = train_index),
                                  tuneGrid = expand.grid(
                                    nrounds = 20,
                                    max_depth = 3,
                                    colsample_bytree = .6,
                                    eta = 0.1,
                                    gamma=0,
                                    min_child_weight = 1,
                                    subsample = .5
                                  ))
xgboostFit

#view Gradient Boosted model
xgboostFit$finalModel

#Neural Network model
nnetFit <- Zoo_train %>% train(type ~ .,
                               method = "nnet",
                               data = .,
                               tuneLength = 5,
                               trControl = trainControl(method = "cv", indexOut = train_index),
                               trace = FALSE)
nnetFit

#view Neural Net model
nnetFit$finalModel

#Comparing all models

resamps <- resamples(list(
  ctree = ctreeFit,
  C45 = C45Fit,
  SVM = svmFit,
  KNN = knnFit,
  rules = rulesFit,
  randomForest = randomForestFit,
  xgboost = xgboostFit,
  NeuralNet = nnetFit
))
resamps

#summary of comparisons
summary(resamps)

#visualization of comparisons with boxplot
bwplot(resamps, layout = c(3, 1))

#Choosing a model to test our data (Random Forest chosen)
pr <- predict(randomForestFit, Zoo_test)
pr

#check the confusion matrix
confusionMatrix(pr, reference = Zoo_test$type)

#DEEP LEARNING

#data preparation
X <- Zoo_train %>% select(!type) %>% 
  mutate(across(everything(), as.integer)) %>% as.matrix()
head(X)
y <- Zoo_train %>% pull("type") %>% as.integer() %>% `-`(1L) %>% to_categorical()
head(y)
X_test <- Zoo_test %>% select(!type) %>% 
  mutate(across(everything(), as.integer)) %>% as.matrix()
y_test <- Zoo_test %>% pull("type") %>% as.integer() %>% `-`(1L) %>% to_categorical()

#model training
model <- keras_model_sequential() %>%
  layer_dense(units = 10, activation = 'relu', input_shape = c(ncol(X)),
              kernel_regularizer=regularizer_l2(l=0.01)) %>%
  layer_dense(units = ncol(y), activation = 'softmax') %>%
  compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = 'accuracy')


history <- model %>% fit(
  X, y,
  batch_size = 10,
  epochs = 100,
  validation_split = .2
)

plot(history)

#making predictions
class_labels <- levels(Zoo_train %>% pull(type))

pr <- predict(model, X_test) %>% apply(MARGIN = 1, FUN = which.max)
pr <- factor(pr, labels = class_labels, levels = seq_along(class_labels))

pr

#checking confusion matrix
confusionMatrix(pr, reference = Zoo_test$type)


#Comparing Boundaries of Different Classification Algorithms
decisionplot <- function(model, data, class_var, 
                         predict_type = c("class", "prob"), resolution = 5 * 75) {
  # resolution is set to 75 dpi if the image is rendered  5 inces wide. 
  
  y <- data %>% pull(class_var)
  x <- data %>% dplyr::select(-all_of(class_var))
  
  # resubstitution accuracy
  prediction <- predict(model, x, type = predict_type[1])
  # LDA returns a list
  if(is.list(prediction)) prediction <- prediction$class
  prediction <- factor(prediction, levels = levels(y))
  
  cm <- confusionMatrix(data = prediction, reference = y)
  acc <- cm$overall["Accuracy"]
  
  # evaluate model on a grid
  r <- sapply(x[, 1:2], range, na.rm = TRUE)
  xs <- seq(r[1,1], r[2,1], length.out = resolution)
  ys <- seq(r[1,2], r[2,2], length.out = resolution)
  g <- cbind(rep(xs, each = resolution), rep(ys, time = resolution))
  colnames(g) <- colnames(r)
  g <- as_tibble(g)
  
  ### guess how to get class labels from predict
  ### (unfortunately not very consistent between models)
  cl <- predict(model, g, type = predict_type[1])
  
  # LDA returns a list
  if(is.list(cl)) { 
    prob <- cl$posterior
    cl <- cl$class
  } else
    try(prob <- predict(model, g, type = predict_type[2]))
  
  # we visualize the difference in probability/score between the 
  # winning class and the second best class.
  # don't use probability if predict for the classifier does not support it.
  max_prob <- 1
  try({
    max_prob <- t(apply(prob, MARGIN = 1, sort, decreasing = TRUE))
    max_prob <- max_prob[,1] - max_prob[,2]
  }, silent = TRUE) 
  
  cl <- factor(cl, levels = levels(y))
  
  g <- g %>% add_column(prediction = cl, probability = max_prob)
  
  ggplot(g, mapping = aes_string(
    x = colnames(g)[1],
    y = colnames(g)[2])) +
    geom_raster(mapping = aes(fill = prediction, alpha = probability)) +
    geom_contour(mapping = aes(z = as.numeric(prediction)), 
                 bins = length(levels(cl)), size = .5, color = "black") +
    geom_point(data = data, mapping =  aes_string(
      x = colnames(data)[1],
      y = colnames(data)[2],
      shape = class_var), alpha = .7) + 
    scale_alpha_continuous(range = c(0,1), limits = c(0,1), guide = "none") +  
    labs(subtitle = paste("Training accuracy:", round(acc, 2)))
}

#importing iris dataset to use for easier visualization
set.seed(1000)
data(iris)
iris <- as_tibble(iris)

### Three classes (MASS also has a select function)
x <- iris %>% dplyr::select(Sepal.Length, Sepal.Width, Species)
x

#view density 
ggplot(x, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) +  
  stat_density_2d(alpha = .2, geom = "polygon") +
  geom_point()

#KNN Boundaries (1 neighbor)
model <- x %>% knn3(Species ~ ., data = ., k = 1)
decisionplot(model, x, class_var = "Species") + labs(title = "kNN (1 neighbor)")

#Naive Beyes Boundaries
model <- x %>% naiveBayes(Species ~ ., data = .)
decisionplot(model, x, class_var = "Species", predict_type = c("class", "raw")) + labs(title = "Naive Bayes")

#CART Boundaries
model <- x %>% rpart(Species ~ ., data = .)
decisionplot(model, x, class_var = "Species") + labs(title = "CART")

#Random Forest Boundaries
model <- x %>% randomForest(Species ~ ., data = .)
decisionplot(model, x, class_var = "Species") + labs(title = "Random Forest")

#SVM Boundaries
model <- x %>% svm(Species ~ ., data = ., kernel = "linear")
decisionplot(model, x, class_var = "Species") + labs(title = "SVM (linear kernel)")
