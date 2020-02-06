##### classification of breast cancer
# we have 10 variables (all factors) and a binary response: benign versus malign.

library(mlbench)
data(BreastCancer)
dim(BreastCancer)

levels(BreastCancer$Class)

head(BreastCancer)

str(BreastCancer)

# data in matrices

tt = BreastCancer[complete.cases(BreastCancer),2:11]
x = NULL
for(i in seq(9)) x = cbind(x,to_categorical(as.numeric(tt[,i])-1))
y = to_categorical(as.numeric(tt[,10])-1)
head(y)

# set training and test

set.seed(17)
ind <- sample(2, nrow(x), 
              replace = TRUE, prob = c(0.7, 0.3))

x.train = x[ind == 1, ]
y.train = y[ind == 1, ]
x.test = x[ind == 2, ]
y.test = y[ind == 2, ]

##### build the DL model with tree layers of neurons:
# Initialize a sequential model
model <- keras_model_sequential()

# Add layers to model
model %>%
  layer_dense(units = 8, activation = 'relu', input_shape = ncol(x.train)) %>%
  layer_dense(units = 5, activation = 'relu') %>%
  layer_dense(units = ncol(y.train), activation = 'softmax')

summary(model)

# use the adam optimizer

# compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = "adam",
  metrics = 'accuracy'
)

# train the model

history <- model %>% fit(
  x = x.train,
  y = y.train,
  epochs = 50,
  batch_size = 50,
  validation_split = 0.2,
  verbose = 2
)

plot(history)

# validate it on the test set

classes <- model %>% predict_classes(x.test)
table(y.test%*%0:1, classes)

(score <- model %>% evaluate(x.test, y.test))


