library(tm)
library(caret)
library(superml)
library(tensorflow)
library(keras)
library(pROC)
#registerDoMC(cores=detectCores()) 

df<- read.csv("C://Users//Damanpreet//Desktop//Dissertation Submission Folder//Final combined dataset.csv", stringsAsFactors = FALSE)
#glimpse(df)

#set.seed(4563)
df <- df[sample(nrow(df)), ]
#df <- df[sample(nrow(df)), ]
View(df)

#df$claim_label <- todf$claim_label,dtype = "int")
df$polarity <- as.factor(df$polarity)
df$subjectivity <- as.factor(df$subjectivity)
df$source <- as.factor(df$source)

df$polarity <- as.integer(df$polarity)
df$subjectivity <- as.integer(df$subjectivity)
df$source <- as.integer(df$source)

#### Text Pre-processing###

df$claim=gsub("[^a-z]"," ",df$claim,ignore.case = TRUE)
#View(df$claim)
corpus <- Corpus(VectorSource(df$claim))
corpus.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stemDocument) %>%
  tm_map(stripWhitespace)

#View(corpus.clean)


#traincorpus <- Corpus(VectorSource(df$claim))

tokenizer = text_tokenizer(num_words = 20000,) %>% 
  fit_text_tokenizer(df$claim)

mat1=texts_to_matrix(tokenizer,texts = df$claim,mode = "tfidf")
dtm2=as.matrix(mat1)

#dim(dtm2)

#write.csv(as.matrix(dtm1),"testdtm.csv")

df.train <- df[1:6000,]
df.test <- df[6001:8000,]

dtm.train <- dtm2[1:6000,]
dtm.test <- dtm2[6001:8000,]

trainNB <- as.matrix(dtm.train)
testNB <- as.matrix(dtm.test)



y2 = to_categorical(as.logical(df.train$claim_label), 2)

newdata=as.matrix(trainNB,df.train$polarity,df.train$subjectivity,df.train$source)
testdata=as.matrix(testNB,df.test$polarity,df.test$subjectivity, df.test$source)

#### Building the model#####

model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = ncol(newdata),) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 2, activation = "softplus")
model %>% compile( loss = "categorical_crossentropy", optimizer = 'Adam',metrics = "accuracy")
model %>% fit( newdata, y2, epochs = 6, trControl = trainControl(
  method = "cv", 
  number = 10,
  verboseIter = TRUE), verbose = 0)

pred=predict_classes(model,testdata)
op = df.test$claim_label

op <- factor(df.test$claim_label,labels = c('0','1'))


c=confusionMatrix(as.factor(pred),op, positive = "1")
c
ro=roc(df.test$claim_label,pred)
a=auc(ro)
a


########Validation set 1################

dfu<- read.csv("C://Users//Damanpreet//Desktop//Dissertation Submission Folder//Validation set 1.csv", stringsAsFactors = FALSE)
#glimpse(df)

dfu <- dfu[sample(nrow(dfu)), ]
#df <- df[sample(nrow(df)), ]
#glimpse(df)

#df$claim_label <- todf$claim_label,dtype = "int")
dfu$polarity <- as.factor(dfu$polarity)
dfu$subjectivity <- as.factor(dfu$subjectivity)
dfu$source <- as.factor(dfu$source)

dfu$polarity <- as.integer(dfu$polarity)
dfu$subjectivity <- as.integer(dfu$subjectivity)
dfu$source <- as.integer(dfu$source)

#df$source

dfu$claim=gsub("[^a-z]"," ",dfu$claim,ignore.case = TRUE)
#View(df$claim)
corpusu <- Corpus(VectorSource(dfu$claim))
corpusu.clean <- corpusu %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stemDocument) %>%
  tm_map(stripWhitespace)

#View(corpus.clean)


mat1u=texts_to_matrix(tokenizer,texts = dfu$claim,mode = "tfidf")
dtm2u=as.matrix(mat1u)

newdatau=as.matrix(dtm2u,dfu$polarity,dfu$subjectivity, dfu$source)

predu=predict_classes(model,newdatau)

opu = dfu$claim_label
opu <- factor(dfu$claim_label,labels = c('0','1'))

cu=confusionMatrix(as.factor(predu),opu, positive = "1")
cu
rou=roc(dfu$claim_label,predu)
au=auc(rou)
au

##### Validation set 2##########
dfu<- read.csv("C://Users//Damanpreet//Desktop//Dissertation Submission Folder//Validation set 2.csv", stringsAsFactors = FALSE)
#glimpse(df)

dfu <- dfu[sample(nrow(dfu)), ]
#df <- df[sample(nrow(df)), ]
#glimpse(df)

#df$claim_label <- todf$claim_label,dtype = "int")
dfu$polarity <- as.factor(dfu$polarity)
dfu$subjectivity <- as.factor(dfu$subjectivity)
dfu$source <- as.factor(dfu$source)

dfu$polarity <- as.integer(dfu$polarity)
dfu$subjectivity <- as.integer(dfu$subjectivity)
dfu$source <- as.integer(dfu$source)

#df$source

dfu$claim=gsub("[^a-z]"," ",dfu$claim,ignore.case = TRUE)
#View(df$claim)
corpusu <- Corpus(VectorSource(dfu$claim))
corpusu.clean <- corpusu %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stemDocument) %>%
  tm_map(stripWhitespace)

#View(corpus.clean)


mat1u=texts_to_matrix(tokenizer,texts = dfu$claim,mode = "tfidf")
dtm2u=as.matrix(mat1u)

newdatau=as.matrix(dtm2u,dfu$polarity,dfu$subjectivity, dfu$source)

predu=predict_classes(model,newdatau)

opu = dfu$claim_label
opu <- factor(dfu$claim_label,labels = c('0','1'))

cu=confusionMatrix(as.factor(predu),opu, positive = "1")
cu
rou=roc(dfu$claim_label,predu)
au=auc(rou)
au




