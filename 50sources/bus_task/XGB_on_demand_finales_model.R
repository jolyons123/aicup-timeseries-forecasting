#Clear memory

rm(list=ls())

#Close all R plots

dev.off()

#Daten laden 
setwd("C:/Users/Flo/Documents/UNI_M/3. Semester/AICup/ai-cup-2022-ace/50sources/Bus_regio/tmp")

regular <- read.csv("on_demand_travel_mod_wdw_busstops_feat.csv")


#Remove Class 3 if there is NA in Passengers (Works)

regular$class <- ifelse(regular$class == 3 & is.na(regular$Passengers), NA, regular$class) 



#Wetterdaten 

setwd("C:/Users/Flo/Documents/UNI_M/3. Semester/AICup/ai-cup-2022-ace/50sources/Bus_regio/tmp")
wetter <- read.csv2("Dat_Wetter_mod.csv")

wetter$date <- as.Date(wetter$date)


#Package laden 

library("plyr")
library(zoo)
library(forecast)
library(dplyr)
require(xts)
require(stringr)
require(ggplot2)
require(data.table)


#Change Time-Format 

regular$date <- as.Date(regular$date)


#Mergen der Daten 

data <- join(regular, wetter, type = "right", by = c("date","hour"))

#Create new TimeStamp 

data$date <- as.POSIXct(paste(data$date, data$hour), format="%Y-%m-%d %H", tz = "UTC")


#Tage 

data$days <- weekdays(data$date)

#Wochenzahl 

data$week <- week(data$date)

#Monate 

data$month <- month(data$date)

#Change Column Order 

data <-  data %>% relocate(c(days,week,month), .before = date)

#data <- data[,-4]

#Throw Columns 

data <- data[,-c(7)]


#Transformiere f?r Fourier Transformation 

#Tage

data[data$days == "Montag",]$days <- 1
data[data$days == "Dienstag",]$days <- 2
data[data$days == "Mittwoch",]$days <- 3
data[data$days == "Donnerstag",]$days <- 4
data[data$days == "Freitag",]$days <- 5
data[data$days == "Samstag",]$days <- 6
data[data$days == "Sonntag",]$days <- 7


#Monat


data$days <- as.numeric(as.character(data$days))


#Monat

#Fourier-Reihen-Ansatz

#Frequenz f?r Stunden (sin/cos)

data$hour_sin <- sin((2*pi)/as.numeric(data$hour+1))
data$hour_cos <- cos((2*pi)/as.numeric(data$hour+1))

#Frequenz f?r Tage (Sin/Cos)

data$days_sin <- sin((2*pi)/as.numeric(data$days))
data$days_cos <- cos((2*pi)/as.numeric(data$days))

#Frequenz f?r Woche (Sin/Cos )

data$week_sin <- sin((2*pi)/as.numeric(data$week))
data$week_cos <- cos((2*pi)/as.numeric(data$week))

#Frequenz f?r Monat (sin/cos)

data$month_sin <- sin((2*pi)/as.numeric(data$month))
data$month_cos <- cos((2*pi)/as.numeric(data$month))

#Werfe Nicht-Frequenzen raus 

data <- data[,-c(1:3,5)]

#Relocate 

data <-  data %>% relocate(c(hour_sin,hour_cos,days_sin,days_cos,week_sin,week_cos,month_sin,month_cos), .before = Passengers)

data <- data %>% relocate(class,.before=hour_sin) 

#-------------------------------------------------------------------------
# Split in jede Station und trainiere lokal auf jeden Subdatensatz 
#------------------------------------------------------------------------- 

#Restlicher Datensatz
#f <- 13
#k <- sub_dat[((505*f+167*f + 1):(505*f+167*f + 24)),]        


#brand <- on_demand[on_demand[,7]==1,]

alle_stationen <- data.frame()


for(f in 1:13){
  
  slot_training <- data.frame()
  
  slot_forecast <- data.frame()
  
  for(p in 13:62){
    
    sub_dat <- data[data[,p]==1,]
    
    #Training
      sub_dat_ohne_na <- sub_dat[1:(504*f+168*(f-1)),]
      sub_dat_ohne_na <- sub_dat_ohne_na[complete.cases(sub_dat_ohne_na$class),]
    
    #Für Prognose 
    
    
      sub_dat_mit_na <- sub_dat[((505*f+167*(f-1))):(505*f+167*f),]
    
      
    #Merge for the station the Training-Set and Forecast-Set 
    
      slot_training <- rbind(slot_training,sub_dat_ohne_na)
      
      
      slot_forecast <- rbind(slot_forecast,sub_dat_mit_na)
    
    
  }
  
    #Jetzt Modelltraining und danach Forecast 
  
  
  #-------------------------------------------------------------------------
  # Make Prediction for Class 0 
  #------------------------------------------------------------------------- 
  
  #1.Step make pred beetween zero and rest-clas 
  
  class0 <- ifelse(slot_training$class==0,1,0)
  
  library(xgboost)
  
  
  #define predictor and response variables in training set
  train_0_x = data.matrix(slot_training[, -c(1,2,3,12)])
  train_0_y = class0
  
  #Train Modell with PR-Recall 
  
  #define final training and testing sets
  xgb_train = xgb.DMatrix(data = train_0_x, label = train_0_y)
  xgb_test = xgb.DMatrix(data = train_0_x, label = train_0_y)
  
  #Train Model 
  
  watchlist <- list(train = xgb_train, eval = xgb_test)
  
  # Define the parameters for binary classification
  
  mod0 <- xgboost(data =xgb_train , max_depth = 20,  eta = 0.015, nrounds = 650  ,
                  eval_metric = "aucpr" , scale_pos_weight = 1,
                  gamma = 2.5,
                  eval_metric = "auc",
                  objective = "binary:logistic")
  
  
  #Create Prediction-Probs for Class 0 
  
  test_0_x = data.matrix(slot_forecast[, -c(1,2,3,12)])
  
  prob_0 <- predict(mod0,test_0_x)
  
  
  #-------------------------------------------------------------------------
  # Make Prediction for Class 1 
  #------------------------------------------------------------------------- 
  
  class1 <- ifelse(slot_training$class==1,1,0)
  
  #define predictor and response variables in training set
  train_1_x = data.matrix(slot_training[, -c(1,2,3,12)])
  train_1_y = class1
  
  #Train Modell with PR-Recall 
  
  #define final training and testing sets
  xgb_train = xgb.DMatrix(data = train_1_x, label = train_1_y)
  xgb_test = xgb.DMatrix(data = train_1_x, label = train_1_y)
  
  #Train Model 
  
  watchlist <- list(train = xgb_train, eval = xgb_test)
  
  # Define the parameters for binary classification
  
  mod1 <- xgboost(data =xgb_train , max_depth = 20,  eta = 0.03, nrounds = 650 ,
                  eval_metric = "aucpr" , scale_pos_weight = sum(class0) / sum(class1),
                  gamma = 2.5,
                  eval_metric = "auc",
                  objective = "binary:logistic")
  
  
  #Create Prediction-Probs for Class 1
  
  test_1_x = data.matrix(slot_forecast[,  -c(1,2,3,12)])
  
  prob_1 <- predict(mod1,test_1_x)
  
  
  
  #-------------------------------------------------------------------------
  # Make Prediction for Class 2
  #------------------------------------------------------------------------- 
  
  class2 <- ifelse(slot_training$class==2,1,0)
  
  #define predictor and response variables in training set
  train_2_x = data.matrix(slot_training[, -c(1,2,3,12)])
  train_2_y = class2
  
  #Train Modell with PR-Recall 
  
  #define final training and testing sets
  xgb_train = xgb.DMatrix(data = train_2_x, label = train_2_y)
  xgb_test = xgb.DMatrix(data = train_2_x, label = train_2_y)
  
  #Train Model 
  
  watchlist <- list(train = xgb_train, eval = xgb_test)
  
  
  # Define the parameters for binary classification
  
  mod2 <- xgboost(data =xgb_train , max_depth = 20,  eta = 0.03, nrounds = 650 ,
                  eval_metric = "aucpr" , scale_pos_weight = sum(class0) / sum(class2),
                  gamma = 2.5,
                  eval_metric = "auc",
                  objective = "binary:logistic")
  
  
  #Create Prediction-Probs for Class 2 
  
  test_2_x = data.matrix(slot_forecast[, -c(1,2,3,12)])
  
  prob_2 <- predict(mod2,test_2_x)
  
  
  
  
  
  #-------------------------------------------------------------------------
  # Make Prediction for Class 3
  #-------------------------------------------------------------------------
  
  class3 <- ifelse(slot_training$class==3,1,0)
  
  #define predictor and response variables in training set
  train_3_x = data.matrix(slot_training[, -c(1,2,3,12)])
  train_3_y = class3
  
  #Train Modell with PR-Recall 
  
  #define final training and testing sets
  xgb_train = xgb.DMatrix(data = train_3_x, label = train_3_y)
  xgb_test = xgb.DMatrix(data = train_3_x, label = train_3_y)
  
  #Train Model 
  
  watchlist <- list(train = xgb_train, eval = xgb_test)
  
  # Define the parameters for binary classification
  
  
  mod3 <- xgboost(data =xgb_train , max_depth = 20, eta = 0.03, nrounds = 650 ,
                  eval_metric = "aucpr" , scale_pos_weight = sum(class0) / sum(class3),
                  gamma = 2.5,
                  eval_metric = "auc",
                  objective = "binary:logistic")
  
  
  #Create Prediction-Probs for Class 2 
  
  test_3_x = data.matrix(slot_forecast[, -c(1,2,3,12)])
  
  prob_3 <- predict(mod3,test_3_x)
  
  
  
  #-------------------------------------------------------------------------
  # bind the forecast-prob and shortcut them 
  #-------------------------------------------------------------------------
  
  all_prob <- data.frame(prob_3,prob_2,prob_1,prob_0)
  
  #Cut the Prob!
  
  temp3 <- all_prob$prob_3 + 0.05
  temp2 <- all_prob$prob_2 + 0.1
  temp1 <- all_prob$prob_1 - 0.05
  temp0 <- all_prob$prob_0 - 0.35
  
  
  all_prob_cut <- data.frame(temp0,temp1,temp2,temp3) 
  names(all_prob_cut) <- c("Class0","Class1","Class2","Class3")
  
  all_prob_cut <- all_prob_cut %>%  mutate(max = max.col(., ties.method = "last")) 
  
  #Reduce the Count by one because it startet counting by one
  
  class_all <- all_prob_cut$max - 1
  
  slot_forecast$max <- class_all
  
  alle_stationen <- rbind(alle_stationen,slot_forecast)
  
  
}

t <- right_join(alle_stationen,regular,by = "X")

s <- data.frame(t$X,t$max,t$Passengers.y)


t <- t %>% 
  mutate(Passengers = coalesce(max,Passengers.y))


###Change data frame to needed solution 

k <- t[,c("date.y","EZone","hour","Passengers")]

k <- k %>% 
  rename(
    date = date.y
  )


##Save File 

setwd("C:/Users/Max Weber/Desktop/Ergebnisse/ErgebnisseNeu")
write.csv(k, file = "regular_station_slot_forecast_tuning_neuste.csv",row.names = FALSE)

