# Read file format
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf

File_object = open(r"./Dataset/test/neg/0_2.txt", "r")

file_data = File_object.read()

print("File Data\n",file_data, "\n")

### Split all the words

data_splited = word_tokenize(file_data)

## Start to pré-processing data

### remove stop words
stopWords = set(stopwords.words('english')) 
wordsFiltered = []

for w in data_splited:
	if w not in stopWords:
		wordsFiltered.append(w)


### Lemmanize words
data_lemmatized = []
lemmatizer = WordNetLemmatizer()

for i in range(len(wordsFiltered)) :
    data_lemmatized.append(lemmatizer.lemmatize(wordsFiltered[i]))

print("Data_lemmatized\n",data_lemmatized)
### Tried to use Stemmer, but it ruined most of the words, even that i got the simplest one 
b = []
porter=PorterStemmer()
for i in range(len(data_lemmatized)) :
	b.append(porter.stem(data_lemmatized[i]))
print("\n",b)


## Caracterize words (Bag of words - BOW)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data_lemmatized)   
# Name that were separeted for the train, look that all the names are, by default, in lowercase and in alphabetical order
names = vectorizer.get_feature_names()

# Search for how many words are repeated 
data_lemmatized = X.toarray()	
# Sum every word
sum_array = []
for i in range(len(names)):
    sum_index = 0
    for j in range(len(data_lemmatized)):
        sum_index += data_lemmatized[j][i]
    sum_array.append(sum_index)

print("\nBOW\n",sum_array)

File_object.close()
