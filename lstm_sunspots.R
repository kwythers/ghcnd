####################################################################################################
#
#                               Forcasting comparisons
#                               LSTM vs...
#                               Kirk R Wythers 2019.03.15 
#
####################################################################################################
##### LSTM to predict sunspots #####

# # Install Keras if you have not installed before
# install_keras()

# load libraries
# Core Tidyverse
library(tidyverse)
library(glue)
library(forcats)

# Time Series
library(timetk)
library(tidyquant)
library(tibbletime)

# Visualization
library(cowplot)

# Preprocessing
library(recipes)

# Sampling / Accuracy
library(rsample)
library(yardstick) 

# Modeling
# library(reticulate)
library(keras)

##### sunspot.month, is available for all of us (it ships with base R) - it’s a ts class (not tidy), so we’ll 
##### convert to a tidy data set using the tk_tbl() function from timetk - we use this instead of as.tibble() 
##### from tibble to automatically preserve the time series index as a zoo yearmon index - last, we’ll convert 
##### the zoo index to date using lubridate::as_date() (loaded with tidyquant) and then change to a tbl_time object 
##### to make time series operations easier
# load the data
sun_spots <- datasets::sunspot.month %>%
  tk_tbl() %>%
  mutate(index = as_date(index)) %>%
  as_tbl_time(index = index)

sun_spots

##### Exploratory Data Analysis #####
##### Visualizing Sunspot Data With Cowplot #####
##### use ggplots and combine them using cowplot::plot_grid() - for the zoomed in plot, we make use of 
##### tibbletime::time_filter(), which is an easy way to perform time-based filtering
p1 <- sun_spots %>%
  ggplot(aes(index, value)) +
  geom_point(color = palette_light()[[1]], alpha = 0.5) +
  theme_tq() +
  labs(
    title = "From 1749 to 2013 (Full Data Set)"
  )

p2 <- sun_spots %>%
  filter_time("start" ~ "1800") %>%
  ggplot(aes(index, value)) +
  geom_line(color = palette_light()[[1]], alpha = 0.5) +
  geom_point(color = palette_light()[[1]]) +
  geom_smooth(method = "loess", span = 0.2, se = FALSE) +
  theme_tq() +
  labs(
    title = "1749 to 1759 (Zoomed In To Show Changes over the Year)",
    caption = "datasets::sunspot.month"
  )

p_title <- ggdraw() + 
  draw_label("Sunspots", size = 18, fontface = "bold", 
             colour = palette_light()[[1]])

plot_grid(p_title, p1, p2, ncol = 1, rel_heights = c(0.1, 1, 1))

##### Evaluating the acf #####
##### determine whether or not an LSTM model may be a good approach - the LSTM will leverage autocorrelation 
##### to generate sequence predictions - our goal is to produce a 10-year forecast using batch forecasting 
##### (a technique for creating a single forecast batch across the forecast region, which is in contrast to a 
##### single-prediction that is iteratively performed one or several steps into the future) - the batch prediction 
##### will only work if the autocorrelation used is beyond ten years - let’s inspect

##### first, we need to review the Autocorrelation Function (ACF), which is the correlation between the time series 
##### of interest in lagged versions of itself - the acf() function from the stats library returns the ACF values for 
##### each lag as a plot - however, we’d like to get the ACF values as data so we can investigate the underlying 
##### data - to do so, we’ll create a custom function, tidy_acf(), to return the ACF values in a tidy tibble

# create the function
tidy_acf <- function(data, value, lags = 0:20) {
  
  value_expr <- enquo(value)
  
  acf_values <- data %>%
    pull(value) %>%
    acf(lag.max = tail(lags, 1), plot = FALSE) %>%
    .$acf %>%
    .[,,1]
  
  ret <- tibble(acf = acf_values) %>%
    rowid_to_column(var = "lag") %>%
    mutate(lag = lag - 1) %>%
    filter(lag %in% lags)
  
  return(ret)
}


# next, test the function out to make sure it works as intended - the function takes our tidy time series, 
# extracts the value column, and returns the ACF values along with the associated lag in a tibble format - we 
# have 601 autocorrelation (one for the time series and it’s 600 lags) - all looks good
max_lag <- 12 * 50

sun_spots %>%
  tidy_acf(value, lags = 0:max_lag)

# plot the ACF with ggplot2 to determine if a high-autocorrelation lag exists beyond 10 years
sun_spots %>%
  tidy_acf(value, lags = 0:max_lag) %>%
  ggplot(aes(lag, acf)) +
  geom_segment(aes(xend = lag, yend = 0), color = palette_light()[[1]]) +
  geom_vline(xintercept = 120, size = 3, color = palette_light()[[2]]) +
  annotate("text", label = "10 Year Mark", x = 130, y = 0.8, 
           color = palette_light()[[2]], size = 6, hjust = 0) +
  theme_tq() +
  labs(title = "ACF: Sunspots")
##### good news - we have autocorrelation in excess of 0.5 beyond lag 120 (the 10-year mark) - we can theoretically 
##### use one of the high autocorrelation lags to develop an LSTM model.
sun_spots %>%
  tidy_acf(value, lags = 115:135) %>%
  ggplot(aes(lag, acf)) +
  geom_vline(xintercept = 120, size = 3, color = palette_light()[[2]]) +
  geom_segment(aes(xend = lag, yend = 0), color = palette_light()[[1]]) +
  geom_point(color = palette_light()[[1]], size = 2) +
  geom_label(aes(label = acf %>% round(2)), vjust = -1,
             color = palette_light()[[1]]) +
  annotate("text", label = "10 Year Mark", x = 121, y = 0.8, 
           color = palette_light()[[2]], size = 5, hjust = 0) +
  theme_tq() +
  labs(title = "ACF: Sunspots",
       subtitle = "Zoomed in on Lags 115 to 135")
##### upon inspection, the optimal lag occurs at lag 125 - this isn’t necessarily the one we will use since we 
##### have more to consider with batch forecasting with a Keras LSTM - with that said, here’s how you can filter() 
##### to get the best lag
optimal_lag_setting <- sun_spots %>%
  tidy_acf(value, lags = 115:135) %>%
  filter(acf == max(acf)) %>%
  pull(lag)

optimal_lag_setting

##### Backtesting: time series cross validation #####
##### when doing cross validation on sequential data, the time dependencies on preceding samples must be preserved - we can 
##### create a cross validation sampling plan by offsetting the window used to select sequential sub-samples - in essence, 
##### we’re creatively dealing with the fact that there’s no future test data available by creating multiple synthetic 
##### “futures” - a process often, esp. in finance, called “backtesting”

##### the rsample package includes facitlities for backtesting on time series - the vignette, “Time Series Analysis Example”, 
##### describes a procedure that uses the rolling_origin() function to create samples designed for time series cross 
##### validation - use this approach

##### Developing a backtesting strategy #####
##### the sampling plan we create uses 50 years (initial = 12 x 50 samples) for the training set and ten years 
##### (assess = 12 x 10) for the testing (validation) set we select a skip span of twenty years (skip = 12 x 20) to 
##### evenly distribute the samples into 11 sets that span the entire 265 years of sunspots history - last, we select 
##### cumulative = FALSE to allow the origin to shift which ensures that models on more recent data are not given an 
##### unfair advantage (more observations) that those on less recent data - the tibble return contains the 
##### rolling_origin_resamples
periods_train <- 12 * 50
periods_test  <- 12 * 10
skip_span     <- 12 * 20

rolling_origin_resamples <- rolling_origin(
  sun_spots,
  initial    = periods_train,
  assess     = periods_test,
  cumulative = FALSE,
  skip       = skip_span
)

rolling_origin_resamples

##### visualizing the backtesting stratigy
# visualize the resamples with two custom functions - the first, plot_split(), plots one of the resampling 
# splits using ggplot2 - note that an expand_y_axis argument is added to expand the date range to the full sun_spots 
# dataset date range - this will become useful when we visualize all plots together

# plotting function for a single split
plot_split <- function(split, expand_y_axis = TRUE, alpha = 1, size = 1, base_size = 14) {
  
# manipulate data
  train_tbl <- training(split) %>%
    add_column(key = "training") 
  
  test_tbl  <- testing(split) %>%
    add_column(key = "testing") 
  
  data_manipulated <- bind_rows(train_tbl, test_tbl) %>%
    as_tbl_time(index = index) %>%
    mutate(key = fct_relevel(key, "training", "testing"))
  
# collect attributes
  train_time_summary <- train_tbl %>%
    tk_index() %>%
    tk_get_timeseries_summary()
  
  test_time_summary <- test_tbl %>%
    tk_index() %>%
    tk_get_timeseries_summary()
  
# visualize
  g <- data_manipulated %>%
    ggplot(aes(x = index, y = value, color = key)) +
    geom_line(size = size, alpha = alpha) +
    theme_tq(base_size = base_size) +
    scale_color_tq() +
    labs(
      title    = glue("Split: {split$id}"),
      subtitle = glue("{train_time_summary$start} to {test_time_summary$end}"),
      y = "", x = ""
    ) +
    theme(legend.position = "none") 
  
  if (expand_y_axis) {
    
    sun_spots_time_summary <- sun_spots %>% 
      tk_index() %>% 
      tk_get_timeseries_summary()
    
    g <- g +
      scale_x_date(limits = c(sun_spots_time_summary$start, 
                              sun_spots_time_summary$end))
  }
  
  return(g)
}

##### the plot_split() function takes one split (in this case Slice01), and returns a visual of the sampling 
##### strategy - we expand the axis to the range for the full dataset using expand_y_axis = TRUE
rolling_origin_resamples$splits[[1]] %>%
  plot_split(expand_y_axis = TRUE) +
  theme(legend.position = "bottom")

##### the second function, plot_sampling_plan(), scales the plot_split() function to all of the samples using 
##### purrr and cowplot
# plotting function that scales to all splits 
plot_sampling_plan <- function(sampling_tbl, expand_y_axis = TRUE, 
                               ncol = 3, alpha = 1, size = 1, base_size = 14, 
                               title = "Sampling Plan") {
  
# map plot_split() to sampling_tbl
  sampling_tbl_with_plots <- sampling_tbl %>%
    mutate(gg_plots = map(splits, plot_split, 
                          expand_y_axis = expand_y_axis,
                          alpha = alpha, base_size = base_size))
  
# make plots with cowplot
  plot_list <- sampling_tbl_with_plots$gg_plots 
  
  p_temp <- plot_list[[1]] + theme(legend.position = "bottom")
  legend <- get_legend(p_temp)
  
  p_body  <- plot_grid(plotlist = plot_list, ncol = ncol)
  
  p_title <- ggdraw() + 
    draw_label(title, size = 18, fontface = "bold", colour = palette_light()[[1]])
  
  g <- plot_grid(p_title, p_body, legend, ncol = 1, rel_heights = c(0.05, 1, 0.05))
  
  return(g)
  
}

# visualize the ENTIRE backtesting strategy with plot_sampling_plan() - we can see how the sampling plan shifts 
# the sampling window with each progressive slice of the train/test splits
rolling_origin_resamples %>%
  plot_sampling_plan(expand_y_axis = TRUE, ncol = 3, alpha = 1, size = 1, base_size = 10, 
                     title = "Backtesting Strategy: Rolling Origin Sampling Plan")

# set expand_y_axis = FALSE to zoom in on the samples
rolling_origin_resamples %>%
  plot_sampling_plan(expand_y_axis = FALSE, ncol = 3, alpha = 1, size = 1, base_size = 10, 
                     title = "Backtesting Strategy: Zoomed In")
##### use this backtesting strategy (6 samples from one time series each with 50/10 split in years and a ~20 year offset) 
##### when testing the veracity of the LSTM model on the sunspots dataset

##### Modeling The Keras Stateful LSTM Model #####
# to begin, develop a single Keras Stateful LSTM model on a single sample from the Backtesting Strategy - we’ll then 
# scale the model to all samples to investigate/validate the modeling performance

# for the single LSTM model, we’ll select and visualize the split for the most recent time sample/slice (Slice11) - the 
# 11th split contains the most recent data
split    <- rolling_origin_resamples$splits[[11]]
split_id <- rolling_origin_resamples$id[[11]]

# reuse the plot_split() function to visualize the split - set expand_y_axis = FALSE to zoom in on the subsample
plot_split(split, expand_y_axis = FALSE, size = 0.5) +
  theme(legend.position = "bottom") +
  ggtitle(glue("Split: {split_id}"))

##### Data Setup #####
# first, combine the training and testing data sets into a single data set with a column key that specifies what set they 
# came from (either “training” or “testing)” - note that the tbl_time object will need to have the index re-specified 
# during the bind_rows() step, but this issue should be corrected in dplyr soon (please upvote it so RStudio focuses on it)
df_trn <- training(split)
df_tst <- testing(split)

df <- bind_rows(
  df_trn %>% add_column(key = "training"),
  df_tst %>% add_column(key = "testing")
  ) %>% 
  as_tbl_time(index = index)

head(df)
tail(df)

##### Preprocessing With Recipes #####
# the LSTM algorithm requires the input data to be centered and scaled - we can preprocess the data using the 
# recipes package - we’ll use a combination of step_sqrt to transform the data and reduce the presence of outliers 
# and step_center and step_scale to center and scale the data - the data is processed/transformed using the bake() 
# function
rec_obj <- recipe(value ~ ., df) %>%
  step_sqrt(value) %>%
  step_center(value) %>%
  step_scale(value) %>%
  prep()

df_processed_tbl <- bake(rec_obj, df)

df_processed_tbl

# capture the center/scale history so we can invert the center and scaling after modeling - the square-root 
# transformation can be inverted by squaring the inverted center/scale values
center_history <- rec_obj$steps[[2]]$means["value"]
scale_history  <- rec_obj$steps[[3]]$sds["value"]

c("center" = center_history, "scale" = scale_history)

##### LSTM Plan #####
##### we need a plan for building the LSTM. First, a couple PRO TIPS with LSTM’s:

##### Tensor Format - 
##### predictors (X) must be a 3D Array with dimensions: [samples, timesteps, features]: the first dimension is the length 
##### of values, the second is the number of time steps (lags), and the third is the number of predictors (1 if univariateor 
##### n if multivariate)
##### Outcomes/Targets (y) must be a 2D Array with dimensions: [samples, timesteps]: The first dimension is the length of 
##### values and the second is the number of time steps (lags)

##### Training/Testing - 
##### the training and testing length must be evenly divisible (e.g. training length / testing length must be a whole number)

##### Batch Size - 
##### the batch size is the number of training examples in one forward/backward pass of a RNN before a weight update
##### the batch size must be evenly divisible into both the training an testing lengths (e.g. training length / batch size 
##### and testing length / batch size must both be whole numbers)

##### Time Steps -
##### a time step is the number of lags included in the training/testing set - for our example, we use a single lag

##### Epochs - 
##### the epochs are the total number of forward/backward pass iterations
##### typically more improves model performance unless overfitting occurs at which time the validation accuracy/loss 
##### will not improve

##### taking this in, we can come up with a plan - we’ll select a prediction of window 120 months (10 years) or the length 
##### of our test set. The best correlation occurs at 125, but this is not evenly divisible by the forecasting range - we 
##### could increase the forecast horizon, but this offers a minimal increase in autocorrelation - we can select a batch size 
##### of 40 units which evenly divides into the number of testing and training observations - we select time steps = 1, which 
##### is because we are only using one lag - finally, we set epochs = 300, but this will need to be adjusted to balance the 
##### bias/variance tradeoff
# Model inputs
lag_setting  <- 120 # = nrow(df_tst)
batch_size   <- 40
train_length <- 440
tsteps       <- 1
epochs       <- 300

##### 2D And 3D Train/Test Arrays #####
##### we can then setup the training and testing sets in the correct format (arrays) as follows - remember, LSTM’s need 3D 
##### arrays for predictors (X) and 2D arrays for outcomes/targets (y)
# training set
lag_train_tbl <- df_processed_tbl %>%
  mutate(value_lag = lag(value, n = lag_setting)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "training") %>%
  tail(train_length)

x_train_vec <- lag_train_tbl$value_lag
x_train_arr <- array(data = x_train_vec, dim = c(length(x_train_vec), 1, 1))

y_train_vec <- lag_train_tbl$value
y_train_arr <- array(data = y_train_vec, dim = c(length(y_train_vec), 1))

# testing set
lag_test_tbl <- df_processed_tbl %>%
  mutate(
    value_lag = lag(value, n = lag_setting)
  ) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "testing")

x_test_vec <- lag_test_tbl$value_lag
x_test_arr <- array(data = x_test_vec, dim = c(length(x_test_vec), 1, 1))

y_test_vec <- lag_test_tbl$value
y_test_arr <- array(data = y_test_vec, dim = c(length(y_test_vec), 1))

##### Building The LSTM Model
##### we can build an LSTM model using the keras_model_sequential() and adding layers like stacking bricks - we’ll use 
##### two LSTM layers each with 50 units - the first LSTM layer takes the required input shape, which is the [time steps, 
##### number of features] - the batch size is just our batch size. We set the first layer to return_sequences = TRUE 
##### and stateful = TRUE - the second layer is the same with the exception of batch_size, which only needs to be 
##### specified in the first layer, and return_sequences = FALSE which does not return the time stamp dimension (2D 
##### shape is returned versus a 3D shape from the first LSTM) - we tack on a layer_dense(units = 1), which is the 
##### standard ending to a keras sequential model. Last, we compile() using the loss = "mae" and the popular 
##### optimizer = "adam
model <- keras_model_sequential()

model %>%
  layer_lstm(units            = 50, 
             input_shape      = c(tsteps, 1), 
             batch_size       = batch_size,
             return_sequences = TRUE, 
             stateful         = TRUE) %>% 
  layer_lstm(units            = 50, 
             return_sequences = FALSE, 
             stateful         = TRUE) %>% 
  layer_dense(units = 1)

model %>% 
  compile(loss = 'mae', optimizer = 'adam')

model

##### Fitting The LSTM Model #####
# fit our stateful LSTM using a for loop (we do this to manually reset states) - this will take a minute or 
# two for 300 epochs to run - we set shuffle = FALSE to preserve sequences, and we manually reset the states 
# after each epoch using reset_states()
for (i in 1:epochs) {
  model %>% fit(x          = x_train_arr, 
                y          = y_train_arr, 
                batch_size = batch_size,
                epochs     = 1, 
                verbose    = 1, 
                shuffle    = FALSE)
  
  model %>% reset_states()
  cat("Epoch: ", i)
  
}

##### Predicting Using The LSTM Model
##### make predictions on the test set, x_test_arr, using the predict() function - we can retransform our predictions 
##### using the scale_history and center_history, which were previously saved and then squaring the result - finally, 
##### we combine the predictions with the original data in one column using reduce() and a custom time_bind_rows() function
# make Predictions
pred_out <- model %>% 
  predict(x_test_arr, batch_size = batch_size) %>%
  .[, 1] 
# retransform values
pred_tbl <- tibble(
  index   = lag_test_tbl$index,
  value   = (pred_out * scale_history + center_history)^2
  ) 
# combine actual data with predictions
tbl_1 <- df_trn %>%
  add_column(key = "actual")

tbl_2 <- df_tst %>%
  add_column(key = "actual")

tbl_3 <- pred_tbl %>%
  add_column(key = "predict")
# create time_bind_rows() to solve dplyr issue
time_bind_rows <- function(data_1, data_2, index) {
  index_expr <- enquo(index)
  bind_rows(data_1, data_2) %>%
    as_tbl_time(index = !! index_expr)
}
ret <- list(tbl_1, tbl_2, tbl_3) %>%
  reduce(time_bind_rows, index = index) %>%
  arrange(key, index) %>%
  mutate(key = as_factor(key))

ret

##### Assessing Performance Of The LSTM On A Single Split
# use the yardstick package to assess performance using the rmse() function, which returns the 
# root mean squared error (RMSE) - our data is in the long format (optimal format for visualizing 
# with ggplot2), so we’ll create a wrapper function calc_rmse() that processes the data into the 
# format needed for yardstick::rmse()
calc_rmse <- function(prediction_tbl) {
  
  rmse_calculation <- function(data) {
    data %>%
      spread(key = key, value = value) %>%
      select(-index) %>%
      filter(!is.na(predict)) %>%
      rename(
        truth    = actual,
        estimate = predict
        ) %>%
      rmse(truth, estimate) %>% 
      select(.estimate) %>% # select the third column then convert to numeric to match example format
      as.numeric()
    }
  
  safe_rmse <- possibly(rmse_calculation, otherwise = NA)
  
  safe_rmse(prediction_tbl)
  
  }

# inspect the RMSE on the model
calc_rmse(ret)
# calc_rmse(ret)[, 3]
##### The RMSE doesn’t tell us the story - we need to visualize - the RMSE will come in handy in 
##### determining an expected error when we scale to all samples in the backtesting strateg

##### Visualizing The Single Prediction ##### 
##### make a plotting function, plot_prediction(), using ggplot2 to visualize the results for a single sample
# setup single plot function
plot_prediction <- function(data, id, alpha = 1, size = 2, base_size = 14) {
  
  rmse_val <- calc_rmse(data) # added explicit call to column 3 - krw
  
  g <- data %>%
    ggplot(aes(index, value, color = key)) +
    geom_point(alpha = alpha, size = size) + 
    theme_tq(base_size = base_size) +
    scale_color_tq() +
    theme(legend.position = "none") +
    labs(
      title = glue("{id}, RMSE: {round(rmse_val, digits = 1)}"),
      x = "", y = ""
    )
  
  return(g)
}

# test out the plotting function setting the id = split_id, which is “Slice11”
ret %>% 
  plot_prediction(id = split_id, alpha = 0.65) +
  theme(legend.position = "bottom")

##### Backtesting The LSTM On All Eleven Samples #####
##### Once we have the LSTM working for one sample, scaling to all 11 is relatively simple. We just need to 
##### create an prediction function that can be mapped to the sampling plan data contained in rolling_origin_resamples

##### Creating An LSTM Prediction Function
# this step looks intimidating, but it’s actually straightforward - copy the code from Section 5.1 - Single LSTM into a 
# function - make it a safe function, which is a good practice for any long-running mapping processes to prevent a single 
# failure from stopping the entire process
predict_keras_lstm <- function(split, epochs = 300, ...) {
  
  lstm_prediction <- function(split, epochs, ...) {
    
    # 5.1.2 data setup
    df_trn <- training(split)
    df_tst <- testing(split)
    
    df <- bind_rows(
      df_trn %>% add_column(key = "training"),
      df_tst %>% add_column(key = "testing")
    ) %>% 
      as_tbl_time(index = index)
    
    # 5.1.3 preprocessing
    rec_obj <- recipe(value ~ ., df) %>%
      step_sqrt(value) %>%
      step_center(value) %>%
      step_scale(value) %>%
      prep()
    
    df_processed_tbl <- bake(rec_obj, df)
    
    center_history <- rec_obj$steps[[2]]$means["value"]
    scale_history  <- rec_obj$steps[[3]]$sds["value"]
    
    # 5.1.4 LSTM plan
    lag_setting  <- 120 # = nrow(df_tst)
    batch_size   <- 40
    train_length <- 440
    tsteps       <- 1
    epochs       <- epochs
    
    # 5.1.5 train  - test Setup
    lag_train_tbl <- df_processed_tbl %>%
      mutate(value_lag = lag(value, n = lag_setting)) %>%
      filter(!is.na(value_lag)) %>%
      filter(key == "training") %>%
      tail(train_length)
    
    x_train_vec <- lag_train_tbl$value_lag
    x_train_arr <- array(data = x_train_vec, dim = c(length(x_train_vec), 1, 1))
    
    y_train_vec <- lag_train_tbl$value
    y_train_arr <- array(data = y_train_vec, dim = c(length(y_train_vec), 1))
    
    lag_test_tbl <- df_processed_tbl %>%
      mutate(
        value_lag = lag(value, n = lag_setting)
      ) %>%
      filter(!is.na(value_lag)) %>%
      filter(key == "testing")
    
    x_test_vec <- lag_test_tbl$value_lag
    x_test_arr <- array(data = x_test_vec, dim = c(length(x_test_vec), 1, 1))
    
    y_test_vec <- lag_test_tbl$value
    y_test_arr <- array(data = y_test_vec, dim = c(length(y_test_vec), 1))
    
    # 5.1.6 LSTM model
    model <- keras_model_sequential()
    
    model %>%
      layer_lstm(units            = 50, 
                 input_shape      = c(tsteps, 1), 
                 batch_size       = batch_size,
                 return_sequences = TRUE, 
                 stateful         = TRUE) %>% 
      layer_lstm(units            = 50, 
                 return_sequences = FALSE, 
                 stateful         = TRUE) %>% 
      layer_dense(units = 1)
    
    model %>% 
      compile(loss = 'mae', optimizer = 'adam')
    
    # 5.1.7 fitting LSTM
    for (i in 1:epochs) {
      model %>% fit(x          = x_train_arr, 
                    y          = y_train_arr, 
                    batch_size = batch_size,
                    epochs     = 1, 
                    verbose    = 1, 
                    shuffle    = FALSE)
      
      model %>% reset_states()
      cat("Epoch: ", i)
      
    }
    
    # 5.1.8 predict and return tidy data
    # make predictions
    pred_out <- model %>% 
      predict(x_test_arr, batch_size = batch_size) %>%
      .[,1] 
    
    # retransform values
    pred_tbl <- tibble(
      index   = lag_test_tbl$index,
      value   = (pred_out * scale_history + center_history)^2
    ) 
    
    # combine actual data with predictions
    tbl_1 <- df_trn %>%
      add_column(key = "actual")
    
    tbl_2 <- df_tst %>%
      add_column(key = "actual")
    
    tbl_3 <- pred_tbl %>%
      add_column(key = "predict")
    
    # create time_bind_rows() to solve dplyr issue
    time_bind_rows <- function(data_1, data_2, index) {
      index_expr <- enquo(index)
      bind_rows(data_1, data_2) %>%
        as_tbl_time(index = !! index_expr)
    }
    
    ret <- list(tbl_1, tbl_2, tbl_3) %>%
      reduce(time_bind_rows, index = index) %>%
      arrange(key, index) %>%
      mutate(key = as_factor(key))
    
    return(ret)
    
  }
  
  safe_lstm <- possibly(lstm_prediction, otherwise = NA)
  
  safe_lstm(split, epochs, ...)
  
}

# test the custom predict_keras_lstm() function out with 10 epochs - it returns the data in long format 
# with “actual” and “predict” values in the key column
predict_keras_lstm(split, epochs = 10)

#####  Mapping The LSTM Prediction Function Over The 11 Samples ##### (takes about 5 - 10 mins) 
# with the predict_keras_lstm() function in hand that works on one split, we can now map to all 
# samples using a mutate() and map() combo. The predictions will be stored in a “list” column called “predict”
sample_predictions_lstm_tbl <- rolling_origin_resamples %>%
  mutate(predict = map(splits, predict_keras_lstm, epochs = 300))

# we now have the predictions in the column “predict” for all 11 splits
sample_predictions_lstm_tbl

##### Assessing The Backtested Performance #####
# assess the RMSE by mapping the calc_rmse() function to the “predict” column
sample_rmse_tbl <- sample_predictions_lstm_tbl %>%
  mutate(rmse = map_dbl(predict, calc_rmse)) %>%
  select(id, rmse)

sample_rmse_tbl

# plot
sample_rmse_tbl %>%
  ggplot(aes(rmse)) +
  geom_histogram(aes(y = ..density..), fill = palette_light()[[1]], bins = 16) +
  geom_density(fill = palette_light()[[1]], alpha = 0.5) +
  theme_tq() +
  ggtitle("Histogram of RMSE")

##### summarize the RMSE for the 11 slices. PRO TIP: Using the average and standard deviation of the RMSE 
##### (or other similar metric) is a good way to compare the performance of various models
sample_rmse_tbl %>%
  summarize(
    mean_rmse = mean(rmse),
    sd_rmse   = sd(rmse)
  )

##### Visualizing The Backtest Results
# create a plot_predictions() function that returns one plot with the predictions for the entire set of 11 
# backtesting samples
plot_predictions <- function(sampling_tbl, predictions_col, 
                             ncol = 3, alpha = 1, size = 2, base_size = 14,
                             title = "Backtested Predictions") {
  
  predictions_col_expr <- enquo(predictions_col)
  
  # Map plot_split() to sampling_tbl
  sampling_tbl_with_plots <- sampling_tbl %>%
    mutate(gg_plots = map2(!! predictions_col_expr, id, 
                           .f        = plot_prediction, 
                           alpha     = alpha, 
                           size      = size, 
                           base_size = base_size)) 
  
  # Make plots with cowplot
  plot_list <- sampling_tbl_with_plots$gg_plots 
  
  p_temp <- plot_list[[1]] + theme(legend.position = "bottom")
  legend <- get_legend(p_temp)
  
  p_body  <- plot_grid(plotlist = plot_list, ncol = ncol)
  
  
  
  p_title <- ggdraw() + 
    draw_label(title, size = 18, fontface = "bold", colour = palette_light()[[1]])
  
  g <- plot_grid(p_title, p_body, legend, ncol = 1, rel_heights = c(0.05, 1, 0.05))
  
  return(g)
  
}

##### Here’s the result - on a data set that’s not easy to predict
sample_predictions_lstm_tbl %>%
  plot_predictions(predictions_col = predict, alpha = 0.5, size = 1, base_size = 10,
                   title = "Keras Stateful LSTM: Backtested Predictions")

##### Predicting The Next 10 Years #####
###### we can predict the next 10 years by adjusting the prediction function to work with the full data set - a new function, 
##### predict_keras_lstm_future(), is created that predicts the next 120 time steps (or ten years)
predict_keras_lstm_future <- function(data, epochs = 300, ...) {
  
  lstm_prediction <- function(data, epochs, ...) {
    
    # 5.1.2 Data Setup (MODIFIED)
    df <- data
    
    # 5.1.3 Preprocessing
    rec_obj <- recipe(value ~ ., df) %>%
      step_sqrt(value) %>%
      step_center(value) %>%
      step_scale(value) %>%
      prep()
    
    df_processed_tbl <- bake(rec_obj, df)
    
    center_history <- rec_obj$steps[[2]]$means["value"]
    scale_history  <- rec_obj$steps[[3]]$sds["value"]
    
    # 5.1.4 LSTM Plan
    lag_setting  <- 120 # = nrow(df_tst)
    batch_size   <- 40
    train_length <- 440
    tsteps       <- 1
    epochs       <- epochs
    
    # 5.1.5 Train Setup (MODIFIED)
    lag_train_tbl <- df_processed_tbl %>%
      mutate(value_lag = lag(value, n = lag_setting)) %>%
      filter(!is.na(value_lag)) %>%
      tail(train_length)
    
    x_train_vec <- lag_train_tbl$value_lag
    x_train_arr <- array(data = x_train_vec, dim = c(length(x_train_vec), 1, 1))
    
    y_train_vec <- lag_train_tbl$value
    y_train_arr <- array(data = y_train_vec, dim = c(length(y_train_vec), 1))
    
    x_test_vec <- y_train_vec %>% tail(lag_setting)
    x_test_arr <- array(data = x_test_vec, dim = c(length(x_test_vec), 1, 1))
    
    # 5.1.6 LSTM Model
    model <- keras_model_sequential()
    
    model %>%
      layer_lstm(units            = 50, 
                 input_shape      = c(tsteps, 1), 
                 batch_size       = batch_size,
                 return_sequences = TRUE, 
                 stateful         = TRUE) %>% 
      layer_lstm(units            = 50, 
                 return_sequences = FALSE, 
                 stateful         = TRUE) %>% 
      layer_dense(units = 1)
    
    model %>% 
      compile(loss = 'mae', optimizer = 'adam')
    
    # 5.1.7 Fitting LSTM
    for (i in 1:epochs) {
      model %>% fit(x          = x_train_arr, 
                    y          = y_train_arr, 
                    batch_size = batch_size,
                    epochs     = 1, 
                    verbose    = 1, 
                    shuffle    = FALSE)
      
      model %>% reset_states()
      cat("Epoch: ", i)
      
    }
    
    # 5.1.8 Predict and Return Tidy Data (MODIFIED)
    # Make Predictions
    pred_out <- model %>% 
      predict(x_test_arr, batch_size = batch_size) %>%
      .[,1] 
    
    # Make future index using tk_make_future_timeseries()
    idx <- data %>%
      tk_index() %>%
      tk_make_future_timeseries(n_future = lag_setting)
    
    # Retransform values
    pred_tbl <- tibble(
      index   = idx,
      value   = (pred_out * scale_history + center_history)^2
    )
    
    # Combine actual data with predictions
    tbl_1 <- df %>%
      add_column(key = "actual")
    
    tbl_3 <- pred_tbl %>%
      add_column(key = "predict")
    
    # Create time_bind_rows() to solve dplyr issue
    time_bind_rows <- function(data_1, data_2, index) {
      index_expr <- enquo(index)
      bind_rows(data_1, data_2) %>%
        as_tbl_time(index = !! index_expr)
    }
    
    ret <- list(tbl_1, tbl_3) %>%
      reduce(time_bind_rows, index = index) %>%
      arrange(key, index) %>%
      mutate(key = as_factor(key))
    
    return(ret)
    
  }
  
  safe_lstm <- possibly(lstm_prediction, otherwise = NA)
  
  safe_lstm(data, epochs, ...)
  
}

# run predict_keras_lstm_future() on the sun_spots data set
future_sun_spots_tbl <- predict_keras_lstm_future(sun_spots, epochs = 300)

# fianly visualize the forecast with the plot_prediction() function, setting id = NULL - we can use filter_time() to 
# zoom in on the dataset since 1900
future_sun_spots_tbl %>%
  filter_time("1900" ~ "end") %>%
  plot_prediction(id = NULL, alpha = 0.4, size = 1.5) +
  theme(legend.position = "bottom") +
  ggtitle("Sunspots: Ten Year Forecast", subtitle = "Forecast Horizon: 2013 - 2023")


####################################################################################################
##### ARIMA to predict sunspots #####

# load libraries
library(dplyr)
library(tidyr)
library(purrr)
library(broom)
library(ggplot2)
library(forecast)

# put sunspots in an object
sunspots <- sunspots

