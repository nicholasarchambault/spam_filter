#!/usr/bin/env python
# coding: utf-8

# # Building a Spam Filter with the Naive Bayes Algorithm
# by Nicholas Archambault
# 
# The goal of this project is to build a filter that will identify whether or not an incoming SMS message is genuine in order to prevent rampant mobile phone spam. The system will evaluate two key probabilities: the probability that a message is spam given its content, and the probability that the message is NOT spam given its content. Armed with these metrics, the system can accurately classify the message as genuine or spam.
# 
# For each new message, the system will use -- and then update, thanks to the addition of the new message -- numbers of spam and non-spam messages divided by the total number of messages to reveal the probability that the new message is in each category.
# 
# $$ P(Spam|New Message) = \frac{P_{Spam} \cdot P_{New Message|Spam}}{P_{Spam}} $$$$ P(Spam^C|New Message) = \frac{P_{Spam^C} \cdot P_{New Message|Spam^C}}{P_{Spam^C}} $$
#     
# We will create a dictionary of all the words used across all messages, spam and non-spam. By evaluating each word in a new message and incorporating its individual probability of being present in both spam and non-spam messages, the system will be able to formulate a clear and accurate prediction for the message's authenticity and categorize it appropriately. The greater the repetition of words in a new message and the higher the number of likely spam words it contains, the greater the chance it is a spam message. 

# ## Reading Data
# 
# We will first read in and explore the database, compiled by the Machine Learning Repository at University of California: Irvine, of 5,572 SMS messages that have already been classified. In the data, non-spam, genuine messages are denoted 'ham'.

# In[1]:


# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("SMSSpamCollection", sep="\t", header = None, names = ["Label", "SMS"])


# In[3]:


data.shape


# In[4]:


data.head(5)


# In[5]:


# Frequency table
data["Label"].value_counts()*100/5572


# ## Training and Testing Sets
# 
# Previously, we found that ~87% of messages are genuine and ~13% are spam. Next, we'll split our data into two sets: a training set, consisting of 4,458 messages (~80% of data) and a testing set of 1,114 messages (~20%). 

# In[6]:


# Randomize Data
randomized = data.sample(frac=1, random_state=1)

# Split data according to 80th percentile index position
training = randomized.iloc[:round(len(randomized)*.8)].reset_index(drop = True)
testing = randomized.iloc[round(len(randomized)*.8):].reset_index(drop = True)


# In[7]:


training["Label"].value_counts()*100/len(training)


# In[8]:


testing["Label"].value_counts()*100/len(testing)


# Since we randomized the data prior to splitting it, we expect -- and find -- that the proportions of spam and non-spam messages among each of our new datasets are the same as they were for the larger set.

# ## Data Cleaning
# 
# To calculate all the probabilities required by the algorithm, we'll first need to perform a bit of data cleaning to bring the data in a format that will allow us to extract easily all the information we need. This will involve tranforming the data to a uniform state, splitting each message into its individual words, and creating a new dataset with a unique column for each word, which gives a value for the number of instances that word occurs in each individual SMS.

# ### Case and Punctuation
# 
# We start by eliminating all punctuation and converting each word to lower case.

# In[9]:


# Before cleaning
training.head()


# In[10]:


# After cleaning
training["SMS"] = training["SMS"].str.replace("\W", " ").str.lower()
training.head()


# ### Creating Vocabulary
# 
# Next, we'll create a list containing every word in the entire dataset. This can be accomplished by splitting the messages on their whitespace, iterating over each message, and appending each unique word to an empty list, `vocabulary`.

# In[11]:


training["SMS"] = training["SMS"].str.split(" ")


# In[12]:


# Create vocabulary of unique words
vocabulary = []
for row in training["SMS"]:
    for i in row:
        vocabulary.append(i)
        
vocabulary = set(vocabulary)
vocabulary = list(vocabulary)


# In[13]:


len(vocabulary)


# It looks like there are 7,784 unique words across all messages.

# ### Final Training Set
# 
# We can use the vocabulary list just created to make our desired transformations. Creating a dictionary counts of each word's usages across the training dataset allows us to transform this dictionary into a pandas dataframe and achieve the format we need to create the spam filter. 

# In[14]:


# Create index of words
word_counts_per_sms = {unique_word: [0] * len(training['SMS']) for unique_word in vocabulary}

# Increment counts
for index, sms in enumerate(training['SMS']):
    for word in sms:
        word_counts_per_sms[word][index] += 1


# In[15]:


# Transform to dataframe
word_counts_per_sms = pd.DataFrame(word_counts_per_sms)
word_counts_per_sms.head()


# In[16]:


# Merge datasets into single training set
training_clean = pd.concat([training, word_counts_per_sms], axis=1)


# In[17]:


training_clean.head()


# The result of these steps is the cleaned, final training dataset upon which to build the filter.

# ## Calculate Constants
# 
# The Naive Bayes algorithm will need to answer these two probability questions to be able to classify new messages:
# 
# $$ P(Spam | w_1,w_2, ..., w_n) \propto P(Spam) \cdot \prod_{i=1}^{n}P(w_i|Spam) $$$$ P(Ham | w_1,w_2, ..., w_n) \propto P(Ham) \cdot \prod_{i=1}^{n}P(w_i|Ham) $$
# 
# Each `w` corresponds to a word in the message under consideration.
# 
# To calculate `P(wi|Spam)` and `P(wi|Ham)` inside the formulas above, we'll need to use these equations:
# $$ P(w_i|Spam) = \frac{N_{w_i|Spam} + \alpha}{N_{Spam} + \alpha \cdot N_{Vocabulary}} $$$$ P(w_i|Ham) = \frac{N_{w_i|Ham} + \alpha}{N_{Ham} + \alpha \cdot N_{Vocabulary}} $$
# 
# Some of the terms in the four equations above will have the same value for every new message. We can calculate the value of these terms once and avoid doing the computations again when a new messages comes in. Below, we'll use our training set to calculate:
# 
#    * P(Spam) and P(Ham)
#    * NSpam, NHam, NVocabulary
# 
# We'll also use Laplace smoothing and set $\alpha = 1$.

# In[18]:


# Isolate spam and ham messages into their own datasets
spams = training_clean[training_clean["Label"] == "spam"]
hams = training_clean[training_clean["Label"] == "ham"]

#P_Spam
p_spam = len(spams)/len(training_clean)

#P_Ham
p_ham = len(hams)/len(training_clean)

#N_Spam
spam_words_per_message = spams["SMS"].apply(len)
n_spam = spam_words_per_message.sum()

#N_Ham
ham_words_per_message = hams["SMS"].apply(len)
n_ham = ham_words_per_message.sum()

#N_Vocabulary
n_vocabulary = len(vocabulary)

#Laplace smoothing
alpha = 1


# ## Calculate Parameters
# 
# Now that we have the constant terms calculated above, we can move on with calculating the parameters $P(w_i|Spam)$ and $P(w_i|Ham)$. Each parameter will thus be a conditional probability value associated with each word in the vocabulary.
# 
# The parameters are calculated using the formulas:
# $$ P(w_i|Spam) = \frac{N_{w_i|Spam} + \alpha}{N_{Spam} + \alpha \cdot N_{Vocabulary}} $$$$ P(w_i|Ham) = \frac{N_{w_i|Ham} + \alpha}{N_{Ham} + \alpha \cdot N_{Vocabulary}} $$

# In[19]:


#Initiate parameters
spam_dict = {unique_word: 0 for unique_word in vocabulary}
ham_dict = {unique_word: 0 for unique_word in vocabulary}


# In[20]:


# Classify parameters
for word in vocabulary:
    n_word_given_spam = spams[word].sum()
    p_word_given_spam = (n_word_given_spam + alpha)/(n_spam + alpha*n_vocabulary)
    spam_dict[word] = p_word_given_spam
    
    n_word_given_ham = hams[word].sum()
    p_word_given_ham = (n_word_given_ham + alpha)/(n_ham + alpha*n_vocabulary)
    ham_dict[word] = p_word_given_ham


# ## Classifying a New Message
# 
# Having created all parameters, we can start using the filter to classify new messages. The filter can be understood as function that: 
# 
# 
#    * Takes in as input a new message (w1, w2, ..., wn).
#    * Calculates P(Spam|w1, w2, ..., wn) and P(Ham|w1, w2, ..., wn).
#    * Compares the values of P(Spam|w1, w2, ..., wn) and P(Ham|w1, w2, ..., wn), and:
#         * If P(Ham|w1, w2, ..., wn) > P(Spam|w1, w2, ..., wn), then the message is classified as ham.
#         * If P(Ham|w1, w2, ..., wn) < P(Spam|w1, w2, ..., wn), then the message is classified as spam.
#         * If P(Ham|w1, w2, ..., wn) = P(Spam|w1, w2, ..., wn), then the algorithm may request human help.
# 

# In[21]:


import re

def spam_filter(message):
    # Convert message to string and clean
    message = str(message)
    message = re.sub("\W", " ", message)
    message = message.lower().split()
    
    # Calculate probabilities
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham
    
    # Update probabilities give content words
    for word in message:
        if word in spam_dict:
            p_spam_given_message *= spam_dict[word]
        if word in ham_dict:
            p_ham_given_message *= ham_dict[word]
    
    # Classify message
    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_spam_given_message > p_ham_given_message:
        return 'spam'
    else:
        return 'needs human classification'


# For each input message, the filter will calculate the probabilities that it is spam or ham. The message is assigned to the category with the greater probability. Messages with nearly equal probabilities will request human categorization.

# In[22]:


# Trial
spam_filter('WINNER!! This is the secret code to unlock the money: C3421.')


# In[23]:


# Trial
spam_filter("Sounds good, Tom, then see u there")


# ## Measuring Filter Accuracy
# 
# Results for the first two individual trials seem promising, and now we can apply the filter to our training dataset, creating a new column, `predicted` with the function's results for each message.

# In[24]:


testing["predicted"] = testing["SMS"].apply(spam_filter)
testing.head()


# We can write a function that prints the number of correctly-classified and total messages and measures the accuracy of the filter based on this percentage.

# In[25]:


correct = 0
total = len(testing)

for index, row in testing.iterrows():
    if row["Label"] == row["predicted"]:
        correct += 1
        
accuracy = correct/total
print("Correct:", correct)
print("Total:", total)
accuracy = round(100*accuracy, 2)
print("Accuracy: ", accuracy, "%", sep = "")


# Our filter classifies messages with over 98% accuracy, a very high mark. Of the 1,114 new messages that it hadn't seen in training, the filter correctly assigns 1,099.

# ## Conclusion
# 
# This project used the Naive Bayes algorithm to build a spam message filter that classified over 98% of new SMS messages correctly. Next steps to improve on this effort could include examining the 14 incorrectly classified messages to determine what went wrong, and deepening the complexity of the filter by making it case-sensitive.
