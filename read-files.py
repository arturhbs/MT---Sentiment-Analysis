from os import listdir

# Method to read all files in a directory
def load_loc(filename):
    file = open(filename,'r')
    text = file.read()
    file.close()
    return text

# Method that recieve two params: the path of the directory and the vector that will be stored the texts from the files
def load_files_directory(directory, vector):
    for filename in listdir(directory):
        path = directory + '/' + filename
        #call another method that takes the read of files
        vector.append(load_loc(path))

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

# Test to see if the arrays are recieving the correct message;
for i in range(len(sentences_test_neg)):
    print(sentences_train_neg[i], "\n")

