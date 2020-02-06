library(keras)
library(tensorflow)
# install_tensorflow()
# install_keras()
library(igraph)


# simple test that the proper python pieces are in place
to_categorical(0:3)

# iris data set
rm(list=ls())
data(iris)
plot(iris$Petal.Length,
     iris$Petal.Width, col = iris$Species)

head(iris)

##### build an iris specie classifier based on the observed four iris dimensions - this is the usual 
##### classification (prediction) problem so we have to consider a training sample and evaluate the 
##### classifier on a test sample

##### Data in TensorFlow #####
# data are
# matrices “`matrix´´´ of doubles.
# categorical variables need to be codified in dummies: one hot encoding
onehot.species = to_categorical(as.numeric(iris$Species) - 1)
iris = as.matrix(iris[, 1:4])
iris = cbind(iris, onehot.species)

# define training and test data sets with random 'ind' vector set at 70:30 ratio
set.seed(17)
ind <- sample(2, nrow(iris),
              replace = TRUE, prob = c(0.7, 0.3))
iris.training <- iris[ind == 1, 1:4]
iris.test <- iris[ind == 2, 1:4]
iris.trainingtarget <- iris[ind == 1, -seq(4)]
iris.testtarget <- iris[ind == 2, -(1:4)]

##### Model building #####
# initialize the model

model <- keras_model_sequential()

# and suppose to use a very simple one

model %>%
  layer_dense(units = ncol(iris.trainingtarget), activation = 'softmax',
              input_shape = ncol(iris.training))
summary(model)

# this is the model structure
model$inputs
model$outputs

# make a plot
g = graph_from_literal(Sepal.Length:Sepal.Width:Petal.Length:Petal.Width---Species,simplify = TRUE)
layout <- layout_in_circle(g, order = order(degree(g)))
plot(g,layout = layout,vertex.color = c(2,2,2,2,3))
##### in the plot, blue colors stand for input and green ones for output - its analytic representation is the 
##### follows:
##### $$Species_j = act.func(\mathbf{w}_j,\mathbf{x} = (PW,PL,SW,SL)),$$
##### where the activation function is the softmax (the all life logistic):
##### $$act.func(\mathbf{w}_j,\mathbf{x}) = \frac{e^{\mathbf{x}^T\mathbf{w}}}{\sum e^{\mathbf{x}^T\mathbf{w}}}$$
##### which estimates $Pr(Specie = j|\mathbf{x} = (PW,PL,SW,SL))$.

##### Model fitting: fit() and the optimizer -estimation consists in finding the weights $\mathbf{w}$ that 
#####minimizes a loss function. For instance, if the response $Y$ were quantitative, then
#####n$$w = \arg\min \sum_{i = 1}^m(y_i-wx_i)^2,$$
##### whose solution is given by the usual equations of derivatives $w$:
##### $$\frac{\partial \sum_{i = 1}^n(y_i-wx_i)^2}{\partial w} = 0,$$
##### note however, that
##### $$\partial \sum (y_i-wx_i)^2 = \sum \partial (y_i-wx_i)^2,$$
##### (Is parallelizable in batches of samples (of length batch_size), that is
##### $$\sum \partial (y_i-wx_i)^2 = \sum{\partial\sum (y_i-wx_i)^2}$$
##### where $n_l$ is the batch_size.
##### suppose in general a non-analytical loss function (the usual case in more complicated networks)  
##### $Q(w) = \sum_{i = 1}^m(y_i-wx_i)^2,$ and suppose that 
##### $\frac{\partial Q(w)}{\partial w} = 0$ is not available analytically - then we would have to use “Newton-Raphson” 
##### optimizer family (or gradient optimizers) whose best known member in Deep Learning (DL) is the Stochastic 
##### Gradient Descent (SGD):
##### starting form an initial weight $w^{(0)}$ at step $m$:
##### $$w^{(m)} = w^{(m-1)}-\eta\Delta Q_i(w),$$
##### where $\eta>0$ is the Learning Rate: the lower (bigger) $\eta$ is, the more (less) steps are needed to 
##### achieve the optimum with a greater (worse) precision
##### it is stochastic in the sense that the index $i$ of the sample is random (avoids overfitting): 
##### $\Delta Q(w) : = \Delta Q_i(w)$ - this also induces complications when (if) dealing with time series

# using SGD with $\eta = 0.01$ we have to set:

sgd <- optimizer_sgd(lr = 0.01)

##### and then this is plugged in into the model and used afterwards in compilation - once it is established, the 
##### loss function $Q$ (here we use the categorical_crossentropy because the response is a non-binary categorical variable):
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = sgd,
  metrics = 'accuracy'
)

##### we have to train it in epochs (i.e. the $m$ steps above) using a portion of the training sample, 
##### validation_split, to verify eventual overfitting (i.e. the model is fitted and the loss evaluated in that 
##### random part of the sample which is finally not used for training):

history <- model %>% fit(
  x = iris.training,
  y = iris.trainingtarget,
  epochs = 100,
  batch_size = 5,
  validation_split = 0.2,
  verbose = 0
)

# the result of the trained model is
plot(history)

# validation on the test sample:
classes <- model %>% predict_classes(iris.test)
table(iris.testtarget%*%0:2, classes)

# with a validation score

(score <- model %>% evaluate(iris.test, iris.testtarget))


