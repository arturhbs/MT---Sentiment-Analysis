from os import listdir

# Method to read all files in a directory
def load_loc(filename):
    file = open(filename,'r')
    text = file.read()
    file.close()
    return text

# Method that recieve two params: the path of the directory and the vector that will be stored the texts from the files
def load_files_directory(directory, arr):
    for filename in listdir(directory):
        path = directory + '/' + filename
        #call another method that takes the read of files
        arr.append(load_loc(path))

# All the directories names
directory_test_neg = './Dataset/test/neg'
directory_test_pos = './Dataset/test/pos'
directory_train_neg = './Dataset/train/neg'
directory_train_pos = './Dataset/train/pos'

# All the arrays names
sentences_test_neg = []
sentences_test_pos = []
sentences_train_neg = []
sentences_train_pos = []

# Inicializing all vectors
load_files_directory(directory_test_neg,sentences_test_neg)
load_files_directory(directory_test_pos,sentences_test_pos)
load_files_directory(directory_train_neg,sentences_train_neg)
load_files_directory(directory_train_pos,sentences_train_pos)

print("Finish reading dataset\n")
##############################################################################################

######################### PRE-PROCESSING DATA 
###### TOKENIZING

from nltk.tokenize import word_tokenize

# Method that need two params, first need the array of all sentences and the second the array that will store all the splited senteces
def tokenize_words(arr, arr_tokenized):
    for i in range(len(arr)):
        data_split = word_tokenize(arr[i])
        arr_tokenized.append(data_split)

# All the new arrays name
data_split_test_neg = []
data_split_test_pos = []
data_split_train_neg = []
data_split_train_pos = []


# Calling method to all new arrays
# tokenize_words(sentences_test_neg,data_split_test_neg)
# tokenize_words(sentences_test_pos,data_split_test_pos)
# tokenize_words(sentences_train_neg,data_split_train_neg)
tokenize_words(sentences_train_pos,data_split_train_pos)

print("Finish tokenize\n")

###### LOWERCASE WORDS

def lowercase(arr, arr_lower):
    for i in range(len(arr)):
        for w in arr[i]:
            arr_lower.append(w.lower())

# All the new array's name
lowercase_test_neg = []        
lowercase_test_pos = []        
lowercase_train_neg = []        
lowercase_train_pos = []        

# lowercase(data_split_test_neg,lowercase_test_neg)
# lowercase(data_split_test_pos,lowercase_test_pos)
# lowercase(data_split_train_neg,lowercase_train_neg)
lowercase(data_split_train_pos,lowercase_train_pos)

print("Finish Lowercase\n")

###### STOP-WORDS

from nltk.corpus import stopwords

# Method that need two params, first the array that has been tokenized and the second the new array that will store all the words that are different from the stopWords
def stop_words(arr, arr_filteredWords):
    for w  in arr :
        if not w in stopWords:
            arr_filteredWords.append(w)

stopWords = stopwords.words('english')
stopWords.extend(['-', ')', '(', 'The', 'This', '...', '!', '--', 'A','.','?',',', 'I', '/', 'br', '<', '>', 'a', '&', ':', "'s", "''", "'re", "'ve","``"])


# All the new array's name
filtered_word_test_neg = []
filtered_word_test_pos = []
filtered_word_train_neg = []
filtered_word_train_pos = []

# Calling method to all new arrays
# stop_words(lowercase_test_neg, filtered_word_test_neg)
# stop_words(lowercase_test_pos, filtered_word_test_pos)
# stop_words(lowercase_train_neg, filtered_word_train_neg)
stop_words(lowercase_train_pos, filtered_word_train_pos)

print("Finish stop word\n")

###### LEMMATIZE WORDS (No plural)

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Method that recieves two params, first the one 
def lemmatize(arr, arr_lemmatized):
    for i in range(len(arr)):
        arr_lemmatized.append(lemmatizer.lemmatize(arr[i]))

# All the new array's name
lemmatized_word_test_neg = []
lemmatized_word_test_pos = []
lemmatized_word_train_neg = []
lemmatized_word_train_pos = []

# Calling method to all new array
# lemmatize(filtered_word_test_neg, lemmatized_word_test_neg)
# lemmatize(filtered_word_test_pos, lemmatized_word_test_pos)
# lemmatize(filtered_word_train_neg, lemmatized_word_train_neg)
lemmatize(filtered_word_train_pos, lemmatized_word_train_pos)

print("Finish lemmatizing\n")


###### PORTERSTEMMER WORDS

from nltk.stem import PorterStemmer

porter = PorterStemmer()

def porterStemmer(arr, arr_potter):
    for i in range(len(arr)):
        arr_potter.append(porter.stem(arr[i]))

porter_word_test_neg = []        
porter_word_test_pos = []        
porter_word_train_neg = []        
porter_word_train_pos = []   

# porterStemmer(lemmatized_word_test_neg, porter_word_test_neg)     
# porterStemmer(lemmatized_word_test_pos, porter_word_test_pos)     
# porterStemmer(lemmatized_word_train_neg, porter_word_train_neg)     
porterStemmer(lemmatized_word_train_pos, porter_word_train_pos) 

print("Finish PorterStemmer\n")    

# Test to see if the arrays are recieving the correct message;
# print(porter_word_train_pos)


######################### END PRE-PROCESSING DATA 
##############################################################################################
######################### BOW

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(porter_word_train_pos)   
# Name that were separeted for the train, look that all the names are, by default, in lowercase and in alphabetical order

terms = vectorizer.get_feature_names()
freqs = X.sum(axis=0).A1
result = dict(zip(terms, freqs))
print(sorted(result.items(), key = 
             lambda kv:(kv[1], kv[0]), reverse=True ))
# # Search for how many words are repeated 
# count_names = X.toarray()	
# # Sum every word
# sum_array = []
# for i in range(len(names)):
#     sum_index = 0
#     for j in range(len(porter_word_train_pos)):
#         sum_index += porter_word_train_pos[j][i]
#     sum_array.append(sum_index)
