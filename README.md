# CS6120-NaturalLanguageProcessing-Fall21-ToxicCommentClassification

Term Project - Toxic Comment Classification

Team: Leora Fink, Jiaxuan Shang, Yu Feng, Quan Gao

Professor Uzair Ahmad

Northeastern University

# Project Description

We built a multi-headed model that’s capable of detecting different types of toxicity like threats, obscenity, insults, and identity-based hate. We have three different models to predict a probability of each type of toxicity for a specific comment. And all these trained models will help us filter out and eliminate toxic comments and get a more productive and respectful online discussion.

Dataset

We use the Toxic Comment Classification Challenge dataset from Kaggle. It contains a large number of Wikipedia comments which have been labeled by human raters for different toxicities(toxic, severe_toxic, obscene, threat, insult and identity_hate). The training dataset contains 159,571 comments and the testing dataset has 153,164 comments. We use the labels for the test data provided by Kaggle, value of -1 indicates it was not used for scoring.

Text Preprocessing

1. Clean comment
Lowercased the corpus, expanded the contractions. Removed hyperlinks, HTML, non-ASCII, punctuations and stopwords.
2. Tokenization
Used nltk.tokenize package to tokenize each comment.
3. Stemming
Used PorterStemmer for word stemming.
4. Lemmatization with POS Tagging
Used WordNetLemmatizer from nltk. POS Tagging helps to improve the accuracy of lemmatization.

Exploratory Data Analysis:

● Tag Frequency - Here we analyzed the occurrence of different tags. We noticed that the different toxicities are not distributed evenly across all comments. Toxic comments are seen the most followed by obscene comments and insult comments.

● Comments with multiple tags

● Correlation - 
Obscene has the highest correlation with the toxic label.

● Word Cloud - 
We built word clouds to help visualize and to get an idea of the kind of words we are dealing with for each comment category. We have four word clouds, each representing four different kinds of comments: toxic, severe toxic, threatening, and insult.

● TF-IDF - 
implemented via sklearn TfidfVectorizer. It takes the preprocessed data and produces the ngram which will be used to find the top words for a single category. Below shows the top words for each class

● Word2Vec - 
In this part we explored the word2vec feature. Word2Vec is a word embedding technique. It’s a 2-layer neural network that takes in a text and produces a vector space. We used the gensim library and the pre-trained word vectors from the Google News corpus to implement Word2Vec. This would come in handy when we work on other models or use other classifiers in the future.

Our Models
1. Logistic Regression
- We implemented Logistic regression ourselves without using an outside logistic regression
library, and used it to classify which comments are associated with each comment category. We treated each category as a separate single classification problem by running the logistic regression a few times, each round with a different train_y depending on which category we were focussing on in that particular round.

2. LightGBM
- LightGBM is a gradient boosting framework that’s based on decision tree algorithms. After
cleaning all the data, the comment texts are converted to a matrix of token counts. We then implemented the LightGBM model from the LightGBM library, trained and predicted the data for each category separately.

3. LSTM
 - Long-short Term memory, which is effective in memorizing important information.
LSTM works by using a multiple word string to find out which class it belongs to. In this project, we used the LSTM model from Keras library. We built a unidirectional model to train and predict our dataset. The model architecture and the dimensions of each layer is shown as below. “None” is inferred automatically by Keras.
After we got the LSTM model running, we also worked on adding a bidirectional LSTM model, by introducing another layer to the existing LSTM model. As you can see from the model evaluation table, since the unidirectional LSTM is already giving us a great result, having bidirectional LSTM didn’t improve the prediction significantly.

Evaluation methodology and results
1. Standard Accuracy
The proportion of true positive results.
2. AUC Score
Sklearn.metric package is used to calculate the AUC Score. Different from accuracy, it can be interpreted as the probability that the model ranks a random positive example more highly than a random negative example.
