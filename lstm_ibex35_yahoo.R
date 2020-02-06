##### LSTM model #####

##### apply the DL to time series analysis: it is not possible to draw train and test randomly and they must 
##### be random sequences of train and test of length batch_size

# load libraries
library(BatchGetSymbols)
library(plotly)

# get data
# from Yahoo Finance download the IBEX 35 time series on the last 15 years and consider the last 3000 days of trading

tickers <- c('%5EIBEX')
first.date <- Sys.Date() - 360*15
last.date <- Sys.Date()

# YAHOO database query and the ACF of the considered IBEX 35 series
myts <- BatchGetSymbols(tickers = tickers,
                        first.date = first.date,
                        last.date = last.date,
                        cache.folder = file.path(tempdir(),
                                                 'BGS_Cache') ) # cache in tempdir()

print(myts$df.control)

y = myts$df.tickers$price.close
myts = data.frame(index = myts$df.tickers$ref.date, price = y, vol = myts$df.tickers$volume)
myts = myts[complete.cases(myts), ]
myts = myts[-seq(nrow(myts) - 3000), ]
myts$index = seq(nrow(myts))

plot_ly(myts, x = ~index, y = ~price, type = "scatter", mode = "markers", color = ~vol)

acf(myts$price, lag.max = 3000)

##### Training and Testing samples 
# data must be standardized

msd.price = c(mean(myts$price), sd(myts$price))
msd.vol = c(mean(myts$vol), sd(myts$vol))
myts$price = (myts$price - msd.price[1])/msd.price[2]
myts$vol = (myts$vol - msd.vol[1])/msd.vol[2]
summary(myts)

# use the first 2000 days for training and the last 1000 for test - remember that the ratio between the number 
# of train samples and test samples must be an integer number as also the ratio between these two lengths with batch_size
datalags = 10
train = myts[seq(2000 + datalags), ]
test = myts[2000 + datalags + seq(1000 + datalags), ]
batch.size = 50

##### Data for LSTM

# predictor $X$ is a 3D matrix:
# first dimension is the length of the time series
# second is the lag;
# third is the number of variables used for prediction $X$ (at least 1 for the series at a given lag)

# response $Y$ is a 2D matrix:
# first dimension is the length of the time series
# second is the lag

x.train = array(data = lag(cbind(train$price, train$vol), datalags)[-(1:datalags), ], 
                dim = c(nrow(train) - datalags, datalags, 2))
y.train = array(data = train$price[-(1:datalags)], dim = c(nrow(train)-datalags, 1))

x.test = array(data = lag(cbind(test$vol, test$price), datalags)[-(1:datalags), ], 
               dim = c(nrow(test) - datalags, datalags, 2))
y.test = array(data = test$price[-(1:datalags)], dim = c(nrow(test) - datalags, 1))

#####  LSTM model codified with Keras
model <- keras_model_sequential()

model %>%
  layer_lstm(units = 100,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

model %>%
  compile(loss = 'mae', optimizer = 'adam')

model

##### train in 2000 steps - remember: for being the model stateful (stateful = TRUE), which means that 
##### the signal state (the latent part of the model) is trained on the batch of the time series, you need 
##### to manually reset the states (batches are supposed to be independent sequences

for(i in 1:2000) {
  model %>% fit(x = x.train,
                y = y.train,
                batch_size = batch.size,
                epochs = 1,
                verbose = 0,
                shuffle = FALSE) 
  model %>% reset_states() 
}

# the prediction
pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]

plot_ly(myts, x = ~index, y = ~price, type = "scatter", mode = "markers", color = ~vol) %>%
  add_trace(y = c(rep(NA, 2000), pred_out), x = myts$index, name = "LSTM prediction", mode = "lines")

# more on validation:
plot(x = y.test, y = pred_out)


# ########################################
# Some notes on Deep Learning
# 
# A deep learning (DL) model is a neural network with many layers of neurons (Schmidhuber 2015), it is an algorithmic 
# approach rather than probabilistic in its nature, see (Breiman and others 2001) for the merits of both approaches. 
# Each neuron is a deterministic function such that a neuron of a neuron is a function of a function along with an 
# associated weight $w$. Essentially for a response variable $Y_i$ for the unit $i$ and a predictor $X_i$ we have to 
# estimate $Y_i = w_1f_1(w_2f_2(…(w_kf_k(X_i))))$, and the larger $k$ is, the “deeper” is the network. With many stacked 
# layers of neurons all connected (a.k.a. dense layers) it is possible to capture high non-linearities and all 
# interactions among variables. The approach to model estimation underpinned by a DL model is that of composition 
# function against that od additive function underpinned by the usual regression techniques including the most 
# modern one (i.e. $Y_i = w_1f_1(X_i)+w_2f_2(X_i)+…+w_kf_k(X_i)$). A thorough review of DL can be found at 
# (Schmidhuber 2015).
# 
# Likely the DL model can be also interpreted as a maximum a posteriori estimation of 
# $Pr(Y|X,Data)$ (Polson, Sokolov, and others 2017) for Gaussian process priors. Despite this and because of its
# complexity it cannot be evaluated the whole distribution $Pr(Y|X,Data)$, but only its mode.
# 
# Estimating a DL consists in just estimating the vectors $w_1,\ldots,w_k$. The estimation requires to evaluate 
# a multidimensional gradient which is not possible to be evaluated jointly for all observations, because of its 
# dimensionality and complexity. Recalling that the derivative of a composite function is defined as the product 
# of the derivative of inner functions (i.e. the chain rule $(f\circ g)’ = (f’\circ g)\cdot g’$) which is implemented 
# for purposes of computational feasibility as a tensor product. Such tensor product is evaluated for batches of 
# observations and it is implemented in the open source software known as Google Tensor Flow 
# ########################################