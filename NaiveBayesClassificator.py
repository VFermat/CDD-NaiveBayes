# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 09:24:56 2018

@author: Vitor Eller
"""

import numpy as np
import pandas as pd
from math import log
import nltk

class NaiveBayesClassificator:
    
    # Initializing the classificator, which can be customized
    def __init__(self, n_gram=1, stem=True, stop_words=True, alpha=1, class_prob=None):
        # Variable that defines the size of the gram to be used
        self.n_gram = n_gram
        # Boolean that defines if the words will be stemmized or not
        self.stem = stem
        # Boolean that defines if stop_words will be removed
        self.stop_words = stop_words
        # Variable that defines which alpha will be used for the Smoothing of probabilities.
        self.alpha = alpha
        """
        class-prob defines how the probability of a specific classification (P(class)) will be calculated
        if None -> P(category) = (number of appearances of the category)/(total)
        if "equal" -> P(category) = 1/(number of categories)
        """
        self.class_prob = class_prob
        # Stemmer that is used if words are to be stemmed
        self.stemmer = nltk.stem.RSLPStemmer()
        # List with the stop-words
        self.stop_words_list = nltk.corpus.stopwords.words('portuguese')
    
    # This function is utilized to clean the sentence
    def _clean_sentence(self, sentence):
        
        # Error occurred while testing. String created to avoid errors
        string = str(sentence)
    
        # Cleaning unwanted characters on the sentence
        string = string.replace(":", " ")
        string = string.replace(";", " ")
        string = string.replace(",", " ")
        string = string.replace("?", " ")
        string = string.replace("(", " ")
        string = string.replace(")", " ")
        string = string.replace("\n", " ")
        string = string.replace("'", " ")
        string = string.replace(".", " ")
        string = string.replace('"', " ")
        string = string.replace("!", " ")
        string = string.replace("@", " ")
        #l Lowercasing all letters, avoids comparisson.
        string = string.lower()

        # Converting the sentence into a list of words
        string_list = []
        
        for word in string.split():
            # Checks if the classifier should remove stop-words
            if self.stop_words:
                # Checks if the word is a stop-word
                if word not in self.stop_words_list:
                    # Checks if the classifier should stemmize the words
                    if self.stem:
                        string_list.append(self.stemmer.stem(word))
                    else:
                        string_list.append(word)
            else:
                # Checks if the classifier should stemmize the words
                if self.stem:
                    string_list.append(self.stemmer.stem(word))
                else:
                    string_list.append(word)
                    
        
        # Returns the n-gram list of the words
        return self._create_gram(string_list)

    # Function that creates the n-gram list of the words
    def _create_gram(self, words_list):
        bigram = []
        # The self.n_gram variable defines how many words will be linked together
        for n in range(len(words_list) + 1 - self.n_gram):
            bigram.append(" ".join(words_list[n:n+self.n_gram]))
        return bigram

    # Function that create a dictionary with the words that appear in a sentence, and its frequencies
    def _create_dict(self, sentences_series):

        
        count = {}

        for sentence in sentences_series:
            # Cleans sentence prior to counting the frequencies
            words_list = self._clean_sentence(sentence)
            for word in words_list:
                if word in count:
                    count[word] += 1
                else:
                    count[word] = 1
                        
        return count
    
    # Function that calculates the d, variable used on the Smoothing of the Probability
    def _get_d(self, df, x_label):
        
        words = []
        for sentence in df[x_label]:
            for word in self._clean_sentence(sentence):
                if word not in words:
                    words.append(word)
                    
        # It returns the total words of the dataset
        return len(words)
    
    # Function that calculates the probability of a specific category
    def _calc_prob(self, sentence, e):
        
        """
        We are using log for the probabilities to get a higher accuracy on the calculation
        If you don't use log:
            P(sentence|category) = P(word1|category)*P(word2|category)*...*P(lastword|category)
        Applying log:
            log(P(sentence|category)) = log(P(word1|category)) + log(P(word2|category)) + ... + log(P(lastword|category))
        """
        
        # Starts the probability with the probability of the specific category
        prob = log(self.classes_dicts[e]["class_prob"])
    
        # Alpha factor for the LaPlace smoothing
        total = self.classes_dicts[e]["n_words"] + self.alpha*self.d
        
        # Calculates the probability for each word (or n-gram) in the cleaned sentence
        for word in self._clean_sentence(sentence):
            if word in self.classes_dicts[e]["words"]:
                count = self.classes_dicts[e]["words"][word] + self.alpha
            else:
                count = self.alpha
            prob += log(count/total)
        
        return prob
    
    # Function that classifies the sentence
    def _classify(self, sentence):
        
        # Variable that stores the highest probability and which category it represents.
        highest = [None, None]
        
        # Calculates the probability for each category
        for e in self.classes:
            classes_probs = self._calc_prob(sentence, e)
            if highest[0] is not None:
                # Checks if the probability for that category is higher than the highest probability 
                if classes_probs > highest[1]:
                    highest[0] = e
                    highest[1] = classes_probs
            # It runs on the first iteration of the for loop. To initialize the 'highest' list
            else:
                highest[0] = e
                highest[1] = classes_probs
                
        # Returns the classification for that sentence
        return highest[0]            
    
    # This function is used to "teach" the classifier based on the training data
    def fit(self, df, x_label, y_label):
        
        self.df = df
        self.x_label = x_label
        self.y_label = y_label
        
        # Stores the possible categories to classify
        self.classes = []
        for e in df[y_label]:
            if e not in self.classes:
                self.classes.append(e)
                
        """
        Creates a dictionary with informations of each category
        keys: values
            "words": dictionary with the informations of words (or n-grams) in the category
            "n-words": number of words (or n-grams) in the category
            "class-prob": probability of that specific category
        """
        self.classes_dicts = {}
        
        # Completes classes_dicts
        for e in self.classes:
            self.classes_dicts[e] = {}
            self.classes_dicts[e]["words"] = self._create_dict(df[df[y_label] == e][x_label])
            self.classes_dicts[e]["n_words"] = len(self.classes_dicts[e]["words"])
            if self.class_prob == None:
                self.classes_dicts[e]["class_prob"] = df[df[y_label] == e][x_label].count()/df[x_label].count()
            elif self.class_prob == "equal":
                self.classes_dicts[e]["class_prob"] = 1/len(self.classes)
            
        # Creates the d (for the smoothing)
        self.d = self._get_d(df, x_label)
            
    # Function used to predict the classification of a specific series
    def predict(self, sentence_series):
        
        # List with the predictions
        predictions = []
        
        # Classify each sentence
        for sentence in sentence_series:
            predictions.append(self._classify(sentence))
            
        # Returns the classifications as a series -> easier to manipulate later with the df
        return pd.Series(predictions)
    
    # Evaluates the classifier performance
    def evaluate(self, y_test, y_pred):
        
        # Count for the correct predictions
        count = 0
        
        # Compares the predictions with the real classifications
        for e in range(len(y_test)):
            if y_test.loc[e] == y_pred.loc[e]:
                count += 1
                
        # Returns a tuple with the Accuracy and the number of correct predictions
        performance = count/(y_test.count())
        return (performance, count)
    
    """
    Creates a confusion_matrix for the classifier predictions
    Problem to be resolved -> does not work with categories that are not numbered and started on 0
    """
    def confusion_matrix(self, y_test, y_pred):
        
        n = [[0] * len(self.classes)] * len(self.classes)
        cm = np.array(n)
        
        for e in range(len(y_test)):
            cls = y_test.loc[e]
            pred = y_pred.loc[e]
            cm[cls][pred] += 1
            
        return cm