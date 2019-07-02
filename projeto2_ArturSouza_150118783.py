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
###### TOKENIZING (slipt data)

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

print("Finish tokenize\n")

###### LOWERCASE WORDS

def lowercase(arr, arr_lower):
    for i in range(len(arr)):
        arr_aux = []
        for w in arr[i]:
            arr_aux.append(w.lower())

        arr_lower.append(arr_aux)

# All the new array's name
lowercase_test_neg = []        
lowercase_test_pos = []        
lowercase_train_neg = []        
lowercase_train_pos = []        

lowercase(data_split_test_neg,lowercase_test_neg)
lowercase(data_split_test_pos,lowercase_test_pos)
lowercase(data_split_train_neg,lowercase_train_neg)
lowercase(data_split_train_pos,lowercase_train_pos)
print("Finish Lowercase\n")
###### STOP-WORDS

from nltk.corpus import stopwords

# Method that need two params, first the array that has been tokenized and the second the new array that will store all the words that are different from the stopWords
def stop_words(arr, arr_filteredWords):
    for i in range(len(arr)):
        arr_aux = []
        for w  in arr[i] :
            if not w in stopWords:
                arr_aux.append(w)
        arr_filteredWords.append(arr_aux)

stopWords = stopwords.words('english')
stopWords.extend(['-', ')', '(', '}' , '{' ,';', '...', '!', '--','.','?',',', '/', 'br', '<', '>', 'a', '&', ':', "'s", "''", "'re", "'ve","``"])


# All the new array's name
filtered_word_test_neg = []
filtered_word_test_pos = []
filtered_word_train_neg = []
filtered_word_train_pos = []

# Calling method to all new arrays
stop_words(lowercase_test_neg, filtered_word_test_neg)
stop_words(lowercase_test_pos, filtered_word_test_pos)
stop_words(lowercase_train_neg, filtered_word_train_neg)
stop_words(lowercase_train_pos, filtered_word_train_pos)

print("Finish stop word\n")

###### LEMMATIZE WORDS (No plural)

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Method that recieves two params, first the original one and the second the array that will be transform
def lemmatize(arr, arr_lemmatized):
    for i in range(len(arr)):
        aux_lemma = []
        for j in arr[i]:
            aux_lemma.append(lemmatizer.lemmatize(j))    
        arr_lemmatized.append(aux_lemma)

# All the new array's name
lemmatized_word_test_neg = []
lemmatized_word_test_pos = []
lemmatized_word_train_neg = []
lemmatized_word_train_pos = []

# Calling method to all new array
lemmatize(filtered_word_test_neg, lemmatized_word_test_neg)
lemmatize(filtered_word_test_pos, lemmatized_word_test_pos)
lemmatize(filtered_word_train_neg, lemmatized_word_train_neg)
lemmatize(filtered_word_train_pos, lemmatized_word_train_pos)

print("Finish lemmatizing\n")


###### PORTERSTEMMER WORDS  It become ambiguous using these and Lemma. 

# from nltk.stem import PorterStemmer

# porter = PorterStemmer()

# def porterStemmer(arr, arr_potter):
#     for i in range(len(arr)):
#         aux_lemma = []
#         for j in arr[i]:
#             aux_lemma.append(porter.stem(j))
#         arr_potter.append(aux_lemma)

# porter_word_test_neg = []        
# porter_word_test_pos = []        
# porter_word_train_neg = []        
# porter_word_train_pos = []   

# porterStemmer(lemmatized_word_test_neg, porter_word_test_neg)     
# porterStemmer(lemmatized_word_test_pos, porter_word_test_pos)     
# porterStemmer(lemmatized_word_train_neg, porter_word_train_neg)     
# porterStemmer(lemmatized_word_train_pos, porter_word_train_pos) 

# print("Finish PorterStemmer")

######################### END PRE-PROCESSING DATA 
##############################################################################################
######################### TRANSFORMING DATA TO BE ABLE TO TRAIN
###### SUM WORDS IN THE LIST

def sumArrayList(arr, sumArray):
    for i in range(len(arr)) :
        final = ''
        for w in arr[i]:
            final = final +' '
            final = final + w
        sumArray.append(final)

sumArray_test_neg = []
sumArray_test_pos = []
sumArray_train_neg = []
sumArray_train_pos = []


sumArrayList(lemmatized_word_test_neg,sumArray_test_neg)
sumArrayList(lemmatized_word_test_pos,sumArray_test_pos)
sumArrayList(lemmatized_word_train_neg,sumArray_train_neg)
sumArrayList(lemmatized_word_train_pos,sumArray_train_pos)

print("Finish Sum Array List")
###### SUM LIST OF TRAINS AND TEST

X_test_final = []
X_train_final = []

X_test_final = sumArray_test_neg + sumArray_test_pos
X_train_final = sumArray_train_neg + sumArray_train_pos

######################### END TRANSFORMING DATA TO BE ABLE TO TRAIN
##############################################################################################
######################### LOGISTIC REGRESSION


from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)
X = cv.fit_transform(X_train_final)
X_test = cv.fit_transform(X_test_final)

target = [1 if i < 12500 else 0 for i in range(25000)]

X_train, X_test, y_train, y_test = train_test_split(
    X, target, train_size = 0.75
)
### Preparing Classifiers
lr = LogisticRegression(C=0.5)
lr.fit(X_train, y_train)
# print ('LOGISTIC REGRESSION - Accuracy for C= : ', accuracy_score(y_test, lr.predict(X_test)))
knn = KNeighborsClassifier(n_neighbors=3,algorithm='kd_tree')
knn.fit(X_train,y_train)
# print("KNN - Accuracy = ", accuracy_score(y_test, knn.predict(X_test)) )
tree = DecisionTreeClassifier()
tree.fit(X_train,y_train)
# print("Decision Tree Classifier - Accuracy = ", accuracy_score(y_test, knn.predict(X_test)) )
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
# print("Random Forest Classifier - Accuracy = ", accuracy_score(y_test, knn.predict(X_test)) )
abc = AdaBoostClassifier()
abc.fit(X_train,y_train)
# print("Ada Boost Classifier - Accuracy = ", accuracy_score(y_test, knn.predict(X_test)) )
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
# print("Gradient Boosting Classifier - Accuracy = ", accuracy_score(y_test, knn.predict(X_test)) )

######################### CROSS VALIDATION

############ Logistic Regression
print("=" * 40)
print("Logistic Regression\n\n")
target_pred = cross_val_predict(lr,X,target,cv=5 )
conf_mat = confusion_matrix(target, target_pred)
print ('\nConfusion matrix : \n',conf_mat)


#Acuracia
print ('\nAcuracia: \n',accuracy_score(target, target_pred))
#Precisão
print ('\nPrecisao: \n',precision_score(target, target_pred)) 
#Revocação
print ('\nRevocacao: \n',recall_score(target, target_pred))
#F1-Score
print ('\nF1-Score: \n',f1_score(target, target_pred))

############ KNN
print("=" * 40)
print("KNN\n\n")
target_pred = cross_val_predict(knn,X,target,cv=5 )
conf_mat = confusion_matrix(target, target_pred)
print ('\nConfusion matrix : \n',conf_mat)


#Acuracia
print ('\nAcuracia: \n',accuracy_score(target, target_pred))
#Precisão
print ('\nPrecisao: \n',precision_score(target, target_pred)) 
#Revocação
print ('\nRevocacao: \n',recall_score(target, target_pred))
#F1-Score
print ('\nF1-Score: \n',f1_score(target, target_pred))

############ Decision Tree
print("=" * 40)
print("Decision Tree Classifier\n\n")
target_pred = cross_val_predict(tree,X,target,cv=5 )
conf_mat = confusion_matrix(target, target_pred)
print ('\nConfusion matrix : \n',conf_mat)


#Acuracia
print ('\nAcuracia: \n',accuracy_score(target, target_pred))
#Precisão
print ('\nPrecisao: \n',precision_score(target, target_pred)) 
#Revocação
print ('\nRevocacao: \n',recall_score(target, target_pred))
#F1-Score
print ('\nF1-Score: \n',f1_score(target, target_pred))

############ Random Forest
print("=" * 40)
print("Random Forest Classifier\n\n")
target_pred = cross_val_predict(rfc,X,target,cv=5 )
conf_mat = confusion_matrix(target, target_pred)
print ('\nConfusion matrix : \n',conf_mat)


#Acuracia
print ('\nAcuracia: \n',accuracy_score(target, target_pred))
#Precisão
print ('\nPrecisao: \n',precision_score(target, target_pred)) 
#Revocação
print ('\nRevocacao: \n',recall_score(target, target_pred))
#F1-Score
print ('\nF1-Score: \n',f1_score(target, target_pred))

############ Ada Boosting
print("=" * 40)
print("Ada Boost Classifier\n\n")
target_pred = cross_val_predict(abc,X,target,cv=5 )
conf_mat = confusion_matrix(target, target_pred)
print ('\nConfusion matrix : \n',conf_mat)


#Acuracia
print ('\nAcuracia: \n',accuracy_score(target, target_pred))
#Precisão
print ('\nPrecisao: \n',precision_score(target, target_pred)) 
#Revocação
print ('\nRevocacao: \n',recall_score(target, target_pred))
#F1-Score
print ('\nF1-Score: \n',f1_score(target, target_pred))

############ Gadient Boosting 
print("=" * 40)
print("Gradient Boosting Classifier\n\n")
target_pred = cross_val_predict(gbc,X,target,cv=5 )
conf_mat = confusion_matrix(target, target_pred)
print ('\nConfusion matrix : \n',conf_mat)


#Acuracia
print ('\nAcuracia: \n',accuracy_score(target, target_pred))
#Precisão
print ('\nPrecisao: \n',precision_score(target, target_pred)) 
#Revocação
print ('\nRevocacao: \n',recall_score(target, target_pred))
#F1-Score
print ('\nF1-Score: \n',f1_score(target, target_pred))

print("=" * 40)

