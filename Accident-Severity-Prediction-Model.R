#Loading libraries
library(rpart,quietly = TRUE)
library(caret,quietly = TRUE)
library(rpart.plot,quietly = TRUE)
library(rattle)
library(lares)
library(kableExtra)
library(tidyverse)
library(randomForest)
library(mlbench)
library(caTools)
library(lares)
library(heatmaply)
library(leaflet.extras)
#For Text Mining
library(tidytext)
#load a few other tidyverse packages for data munging
library(lubridate)
library(scales)
library(purrr)
library(broom)
library(tm)
library(SnowballC)
library(wordcloud)
library(RColorBrewer)
library(rfUtilities)

# Loading the dataset
#ref: https://stackoverflow.com/questions/32556967/understanding-r-is-na-and-blank-cells
USaccidents <- read.csv("Datasets/US_Accidents_Dec21_updated.csv", na.strings = c("", "NA"))
#View(USaccidents)

# Knowing the data set
dim(USaccidents)
str(USaccidents)

# Overview of the data set
tail(USaccidents, 6)
summary(USaccidents)

# checking for duplicate
dim(USaccidents[duplicated(USaccidents$id),])[1]

# Checking for missing and NA values
nrow(na.omit((USaccidents)))
nrow(USaccidents) - sum(complete.cases(USaccidents))

#new england USaccidents
states_NewEngland <- c("RI", "MA", "VT", "ME", "CT", "NH")
NE_USaccidents <- USaccidents %>% filter(State %in% states_NewEngland)
rm(USaccidents)
dim(NE_USaccidents)
#str(NE_USaccidents)


# Finding the unique values in the columns
unique_count <- sapply(lapply(NE_USaccidents, unique), length, simplify=FALSE)
singleValueCols <- data.frame(unique_count[!(unique_count > 1)])
colnames(singleValueCols)

# Removing columns with factor = 1
NE_USaccidents <- select(NE_USaccidents, -all_of(colnames(singleValueCols)))
dim(NE_USaccidents)

#Checking for missing data(NA) percentage in each column
missing_NA <- round((colMeans(is.na(NE_USaccidents)))*100,2)
order <- order(missing_NA,decreasing = TRUE)
missing_NA_ordered <- missing_NA[order]
head(missing_NA_ordered,3)

#Dropping columns with high NA values
drop_Cols <- c("Number", "Precipitation.in.", "Wind_Chill.F." )
NE_USaccidents <- select(NE_USaccidents, -all_of(drop_Cols))
NE_USaccidents <- na.omit(NE_USaccidents)
dim(NE_USaccidents)


#checking correlation btw all numeric variables and Severity
NE_USaccidents_numCols <- NE_USaccidents %>% select_if(is.numeric)
NE_USaccidents_numCols <- cbind(NE_USaccidents$Severity, NE_USaccidents_numCols)
#str(NE_USaccidents_numCols2)
colnames(NE_USaccidents_numCols)[1] <- 'Severity'
NE_USaccidents_numCols <- NE_USaccidents_numCols %>% mutate_if(is.factor, as.numeric)
cor_numCols <- cor(NE_USaccidents_numCols[ ,colnames(NE_USaccidents_numCols) != "Severity"],
                NE_USaccidents_numCols$Severity)
cor_numCols <- as.data.frame(cor_numCols)
colnames(cor_numCols) <- c("r")
cor_numCols <- tibble::rownames_to_column(cor_numCols, "Variables")

arrange(cor_numCols,r) %>% 
  kable(caption = "Correlation of Severity with numeric variables",
        align = "lc",
        table.attr = "style='width:50%;'") %>%
  kable_classic_2(font_size = 14)

#checking correlation btw all boolean variables and Severity
#https://stackoverflow.com/questions/22772279/converting-multiple-columns-from-character-to-numeric-format-in-r
colnames(NE_USaccidents)
NE_USaccidents_facCols <- select(NE_USaccidents,
                                 c(26:42)) %>% mutate_if(is.character,as.factor)
colnames(NE_USaccidents_facCols)
#str(NE_USaccidents_facCols)
NE_USaccidents_FaNum <- NE_USaccidents_facCols %>% mutate_if(is.factor, as.numeric)
#str(NE_USaccidents_FaNum)
NE_USaccidents_numCols2 <- cbind(NE_USaccidents$Severity, NE_USaccidents_FaNum)
#str(NE_USaccidents_numCols2)
colnames(NE_USaccidents_numCols2)[1] <- 'Severity'

x1 <- cor(NE_USaccidents_numCols2[ ,colnames(NE_USaccidents_numCols2) != "Severity"],
          NE_USaccidents_numCols2$Severity)
cor_numCols2 <- as.data.frame(x1)
colnames(cor_numCols2) <- c("r")

cor_numCols2 <- tibble::rownames_to_column(cor_numCols2, "Variables")
arrange(cor_numCols2,r) %>% 
  kable(caption = "Correlation of Severity with other variables",
        align = "lc",
        table.attr = "style='width:50%;'") %>%
  kable_classic_2(font_size = 14)


# Correlation between independent variables
corr_cross(NE_USaccidents,
           max_pvalue = 0.05,
           top = 15 )

# Listing the columns to be dropped based on multicolinearity results
drop_Cols_2 <- c("ID","End_Lat","End_Lng","Start_Lng",
                 "Civil_Twilight","Nautical_Twilight",
                 "Astronomical_Twilight","Start_Lng",
                 "City","State","County","Wind_Direction",
                 "Weather_Timestamp","Description","Street",
                 "Traffic_Calming","Airport_Code",
                 "Weather_Condition","Crossing",
                 "Start_Time","End_Time","Timezone","Zipcode")

# removing the columns from the dataset
NE_USaccidents_1 <- select(NE_USaccidents,-all_of(drop_Cols_2) )
dim(NE_USaccidents_1) #44048 obs. 20 var /43972


#Data Exploration
# Visualizing Severity
barplot(table(NE_USaccidents_1$Severity),
        main = "Severity Analysis of Accidents in New",
        xlab = "Severity",
        ylab = "Count of Incidents",
    
            col = "lightblue")

#PREPARING THE DATA
# Converting Severity into Low and High
NE_USaccidents_1$Severity<- as.factor(ifelse(NE_USaccidents_1$Severity<=2,"Low","High"))


#Temperature, Wind Speed and Humidity distribution for severity level
par(mfrow = c(1, 3))
plot(NE_USaccidents_1$Severity, NE_USaccidents_1$Wind_Speed.mph.,
     xlab= "Accident Severity",
     ylab = "Wind Speed in Miles per hour")
plot(NE_USaccidents_1$Severity, NE_USaccidents_1$Humidity...,
     xlab= "Accident Severity",
     ylab = "Humidity in percentage")
plot(NE_USaccidents_1$Severity, NE_USaccidents_1$Temperature.F.,
     xlab= "Accident Severity",
     ylab = "Temperature in Fahrenheit")



#data splicing
set.seed(12345)
train <- sample(1:nrow(NE_USaccidents_1),
                size = ceiling(0.80*nrow(NE_USaccidents_1)),
                replace = FALSE)

# Creating the training set
NE_USaccidents_train <- NE_USaccidents_1[train,]
dim(NE_USaccidents_train) # 35239 obs. 20 var /35178    20
table(NE_USaccidents_train$Severity)

# Creating the test set
NE_USaccidents_test <- NE_USaccidents_1[-train,]
dim(NE_USaccidents_test) # 8809 obs.  20 var /8794   20
table(NE_USaccidents_test$Severity)


# MODEL - 1
# Classification Tree

# Finding the perfect split
number.perfect.splits <- apply(X=NE_USaccidents_1[-1],
                               MARGIN = 2,
                               FUN = function(col){
                                 t <- table(NE_USaccidents_1$Severity,col)
                                 sum(t == 0)})

# Descending order of perfect splits
order <- order(number.perfect.splits,decreasing = TRUE)
number.perfect.splits <- number.perfect.splits[order]

# Building the classification tree with rpart
rpart_tree <- rpart(Severity~.,
              data = NE_USaccidents_train,
              method = "class")

# Visualize the decision tree with rpart.plot
rpart.plot(rpart_tree, nn=TRUE)

# choosing the best complexity parameter "cp" to prune the tree
cp.optim <- rpart_tree$cptable[which.min(rpart_tree$cptable[,"xerror"]),"CP"]

# tree prunning using the best complexity parameter
rpart_tree_optim <- prune(rpart_tree, cp=cp.optim)

# Visualize the decision tree with rpart.plot
rpart.plot(rpart_tree_optim, nn=TRUE)


#Testing the model
rpart_tree_pred <- predict(object = rpart_tree_optim, 
                           NE_USaccidents_test[-1],
                           type="class")


#Calculating accuracy
acc_rpart_pred <- table(NE_USaccidents_test$Severity, rpart_tree_pred) 
confusionMatrix(acc_rpart_pred) 


# MODEL - 2
#Random Forest model building

#Creating varaibles for RFM
X_train <- NE_USaccidents_train[, !(colnames(NE_USaccidents_train) == 'Severity')]
Y_train <- NE_USaccidents_train[,'Severity']
X_test <- NE_USaccidents_test[, !(colnames(NE_USaccidents_test) == 'Severity')]
Y_test <- NE_USaccidents_test[,'Severity']


#Random Forest modelling for Severity of Accident
set.seed(123)
ne_accidents_rf <- randomForest(x = X_train,
                               y = Y_train,
                               mtry = 12,
                               importance = TRUE,
                               ntree = 500)


#Using the model to predict for the test dataset
randomForest_pred <- predict(ne_accidents_rf, newdata = NE_USaccidents_test)

#Plotting the predicted results
plot(randomForest_pred, NE_USaccidents_test$Severity)
abline(0,1)

#Confusion Matrix
confusionMatrix(randomForest_pred, NE_USaccidents_test$Severity)

#Calculating the test mean value
mean(randomForest_pred == NE_USaccidents_test$Severity)

#Finding the Variable of Importance
importance(ne_accidents_rf)
varImp(ne_accidents_rf)

#Plot MeanDecreaseAccuracy and MeanDecreaseGini
varImpPlot(ne_accidents_rf)



# MODEL - 3
#Gradient boosting model building

#gbm model 1
#Referred from rpubs.com: https://rpubs.com/billantem/651903
GBM_1<- train(Severity~Start_Lat+Distance.mi.+Temperature.F.+ Pressure.in.+Visibility.mi.+Station+
                Sunrise_Sunset + Amenity+Bump+Junction+No_Exit+Humidity...+Side+Give_Way+Roundabout+
                Railway+Traffic_Signal+Wind_Speed.mph.,
              data = NE_USaccidents_train,
              method = "gbm",
              trControl = trainControl(method="CV", repeats = 10),
              preProcess = c("center", "scale"))

summary(GBM_1, las = 1)

#Using the model to predict for the test dataset
GBM_pred_1 <- predict(GBM_1,newdata = NE_USaccidents_test)
confusionMatrix(GBM_pred_1, NE_USaccidents_test$Severity)


#gbm model 2
#Referred from rpubs.com: https://rpubs.com/billantem/651903

GBM_2 <- train(Severity~Start_Lat+Distance.mi.+Temperature.F.+ Station+
                 Sunrise_Sunset + Amenity + Junction +Humidity...+Side+Give_Way+
                 Traffic_Signal+Wind_Speed.mph.,
               data = NE_USaccidents_train,
               method = "gbm",
               trControl = trainControl(method="CV", repeats = 10),
               preProcess = c("center", "scale"))

summary(GBM_2, las = 1)

#Using the model to predict for the test dataset
GBM_pred_2 <- predict(GBM_2,newdata = NE_USaccidents_test)
confusionMatrix(GBM_pred_2, NE_USaccidents_test$Severity)


#Final model - improving RF model
#testing significance
rf.perm <- rf.significance(ne_accidents_rf,
                           NE_USaccidents_train[,1:20],
                           q = 0.99,
                           p = 0.05,
                           nperm = 99,
                           ntree=500)

#creating calculated field - duration
NE_USaccidents <- NE_USaccidents %>%
  mutate(Start_Time = ymd_hms(Start_Time),
         End_Time = ymd_hms(End_Time))
NE_USaccidents <- NE_USaccidents %>%
  mutate(durationMin = round(End_Time - Start_Time))

#Visualizations to support our analysis

#histogram of timestamp
options(scipen=5)
ggplot(NE_USaccidents,
       aes(x = durationMin, fill = Severity)) +
  geom_histogram(position = "identity",
                 bins = 50,
                 show.legend = FALSE) +
  facet_wrap(~Severity, ncol = 1) + xlim(0,500)

#Visualizing the accident hotspot locations
# Converting Severity into Low and High
NE_USaccidents$Severity<- as.factor(ifelse(NE_USaccidents$Severity<=2,"Low","High"))
severity_high <- NE_USaccidents %>% filter(NE_USaccidents$Severity=="High")

#Creating a map
map <- severity_high %>%
  leaflet () %>%
  addTiles () %>%
  addHeatmap(lng=~Start_Lng,
             lat=~Start_Lat,
             intensity=2,
             blur=4,
             max=1,
             radius=4)
map


#Distance distribution for severity level
summary(NE_USaccidents_1$Distance.mi.)
plot(NE_USaccidents_1$Distance.mi.~NE_USaccidents_1$Severity,
     xlab= "Accident Severity",
     ylab = "Distance in Miles")


#Barplot for severity levels in various states (4 levels)
ggplot(NE_USaccidents, aes(fill=Severity, y=Temperature.F., x=State)) +
  geom_bar(position='dodge', stat='identity')



# TEXT MINING
# Load the data as a corpus
docs_ne <- Corpus(VectorSource(NE_USaccidents$Description))

# Convert the text to lower case
docs_ne <- tm_map(docs_ne, content_transformer(tolower))

# Remove english common stopwords
docs_ne <- tm_map(docs_ne, removeWords, stopwords("english"))

# Defining our stop words
myStopwords <- c("exit", "exits", "left", "right", "closed", "blocked",
                 "miles","congested","near","expect","tpke","incident","lane",
                 "lanes","vehicle","due","reported","traffic","ave","near",
                 "between","at","accident","rd","due","road","st","directions",
                 "stationary","route","take","long","shoulder","closed","crash",
                 "center","slow","delays","northbound","southbound","eastbound",
                 "westbound","motor","ctct")

# Removing the defined stop words
docs_ne <- tm_map(docs_ne,
                  removeWords,
                  c("accident|aveexit|exits|near|incident|road|stexit|rdexit|ctexit|lane|motor|usexit|miles|near|expect|tpke|incident|lane|vehicle|due|reported|exits|ave|near|Between|At|Rd|due|Road|St|Directions|lane|lanes|road|center|ctct|main|take"))

# Removing numbers
docs_ne <- tm_map(docs_ne, removeNumbers)

# Removing punctuation
docs_ne <- tm_map(docs_ne, removePunctuation)

# Eliminate extra white spaces
docs_ne <- tm_map(docs_ne, stripWhitespace)

#Viewing the varaible
#inspect(docs_ne)

dtm_ne <- TermDocumentMatrix(docs_ne)
m_ne <- as.matrix(dtm_ne)
v_ne <- sort(rowSums(m_ne),decreasing = TRUE)
d_ne <- data.frame(word = names(v_ne), freq = v_ne)
head(d_ne, 25)

#d_ne$freq

# Using wordcloud to see the most frequent words
set.seed(1234)
wordcloud(words = d_ne$word,
          freq = d_ne$freq,
          min.freq = 1,
          max.words = 200,
          random.order = FALSE,
          rot.per = 0.35, 
          colors = brewer.pal(8, "Dark2"))

findFreqTerms(dtm_ne, lowfreq = 100) #change to 4 if using MLK speech

findAssocs(dtm_ne, terms = "crash", corlimit = 0.1) #change to "freedom" if using MLK speech

barplot(d_ne[100:120,]$freq,
        las = 2,
        names.arg = d_ne[100:120,]$word,
        col = "lightblue",
        main = "Most frequent words",
        ylab = "Word frequencies")



#converting the timestamp
NE_USaccidents_2 <- NE_USaccidents %>%
  mutate(timestamp = ymd_hms(NE_USaccidents$Start_Time))

#histogram of timestamp
ggplot(NE_USaccidents_2,
       aes(x = timestamp, fill = Severity)) +
  geom_histogram(position = "identity",
                 bins = 20,
                 show.legend = FALSE) +
  facet_wrap(~Severity, ncol = 1)


# Compare word frequencies
#list of words to remove
remove_reg1 <- "Exit|Between|At|accident|Rd|closed|due to|Road|St|Directions|Stationary|Accident"

#removing the listed words
tidy_neaccidents <- NE_USaccidents_2 %>% 
  mutate(Description = str_remove_all(Description, remove_reg1))

tidy_neaccidents <- tidy_neaccidents %>% 
  unnest_tokens(word, Description, token = "words")  

tidy_neaccidents <- tidy_neaccidents %>%
  filter(!word %in% stop_words$word,
         !word %in% str_remove_all(stop_words$word, "'"),
         str_detect(word, "[a-z]"))
#View(tidy_neaccidents)

#look at overall frequency of words
frequency <- tidy_neaccidents %>% 
  count(Severity, word, sort = TRUE) %>% 
  left_join(tidy_neaccidents %>% 
              count(Severity, name = "total")) %>%
  mutate(freq = n/total)

frequency

#look at tf-idf instead
neaccidents_IDF <- frequency %>%
  bind_tf_idf(word, Severity, n)

neaccidents_IDF %>%
  select(-total) %>%
  arrange(desc(tf_idf))

#pivot frequency to a wide format with both frequencies as columns
frequency <- frequency %>% 
  select(Severity, word, freq) %>% 
  pivot_wider(names_from = Severity, values_from = freq) %>%
  arrange(High, Low)

frequency

#plot these frequencies
ggplot(frequency, aes(High, Low)) +
  geom_jitter(alpha = 0.1, size = 2.5, width = 0.25, height = 0.25) +
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
  scale_x_log10(labels = percent_format()) +
  scale_y_log10(labels = percent_format()) +
  geom_abline(color = "red")

