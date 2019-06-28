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
tokenize_words(sentences_test_neg,data_split_test_neg)
tokenize_words(sentences_test_pos,data_split_test_pos)
tokenize_words(sentences_train_neg,data_split_train_neg)
tokenize_words(sentences_train_pos,data_split_train_pos)


###### STOP-WORDS

from nltk.corpus import stopwords

# Method that need two params, first the array that has been tokenized and the second the new array that will store all the words that are different from the stopWords
def stop_words(arr, arr_filteredWords):
    for i in range(len(arr)):
        for w in arr[i] :
            arr_filteredWords.append(w)

stopWords = set(stopwords.words('english'))

# All the new arrays name
filtered_word_test_neg = []
filtered_word_test_pos = []
filtered_word_train_neg = []
filtered_word_train_pos = []

# Calling method to all new arrays
stop_words(data_split_test_neg, filtered_word_test_neg)
stop_words(data_split_test_pos, filtered_word_test_pos)
stop_words(data_split_train_neg, filtered_word_train_neg)
stop_words(data_split_train_pos, filtered_word_train_pos)


# Test to see if the arrays are recieving the correct message;
# for i in range(len(filtered_word_test_neg)):
#     print(filtered_word_test_neg[i], "\n")
