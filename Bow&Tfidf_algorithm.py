import nltk

para = '''NLP drives computer programs that translate text from one language to another,
        respond to spoken commands, and summarize large volumes of text rapidly—even in real time.
        There’s a good chance you’ve interacted with NLP in the form of voice-operated GPS systems, 
        digital assistants, speech-to-text dictation software, 
        customer service chatbots, and other consumer conveniences. 
        But NLP also plays a growing role in enterprise solutions that help streamline business operations, 
        increase employee productivity, and simplify mission-critical business processes.
        Natural language processing is an interdisciplinary subfield of computer science and linguistics. 
        It is primarily concerned with giving computers the ability to support and manipulate human language.'''

import re 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps=PorterStemmer()
wordnet = WordNetLemmatizer()
sentences = nltk.sent_tokenize(para)

corpus = []

for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]   
    review = ' '.join(review)
    corpus.append(review)
    
#Bow feature or Algorithm   
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()

#Tfidf feature or Algorithm
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
x_tf = tf.fit_transform(corpus).toarray()





