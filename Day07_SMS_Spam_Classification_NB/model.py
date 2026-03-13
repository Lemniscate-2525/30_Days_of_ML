import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from wordcloud import WordCloud

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Dataset :
file_path = "/content/spam.csv"
df = pd.read_csv(file_path, encoding = 'latin-1')

# Text Preprocessing :
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

df['label'] = df['label'].map({'spam' : 0, 'ham' : 1})

# Text-EDA(NLP like) : 
sns.countplot(x = df['label'])   # Class Dist Ham/Spam : 
plt.title("Class Dist : ")
plt.show()

df['length'] = df['text'].apply(len)  # Message Len Dist : 

sns.histplot(df[df['label'] == 0]['length'], color = 'blue', label = 'Ham', kde = True)
sns.histplot(df[df['label'] == 1]['length'], color = 'red', label = "Spam", kde = True)
plt.legend()
plt.title("Message Len Dist : ")
plt.show()

df['text'] = df['text'].astype(str)  # Clean Dataset 

spam_text = " ".join(df[df['label'] == 1]['text'].tolist())
ham_text =  " ".join(df[df['label'] == 0]['text'].tolist())

print("Spam Len :", len(spam_text))
print("Ham Len :", len(ham_text))

wc_spam = WordCloud(width = 800, height = 400, background_color = 'white', min_font_size = 10).generate(spam_text) # Word Cloud

plt.figure(figsize = (10,5))
plt.imshow(wc_spam)
plt.axis("off")
plt.title("Spam WordCloud : ")
plt.show()

wc_ham = WordCloud(width = 800, height = 400, background_color = 'white', min_font_size = 10).generate(ham_text)

plt.figure(figsize = (10,5))
plt.imshow(wc_ham)
plt.axis("off")
plt.title("Ham WordCloud : ")
plt.show() 


tfidf = TfidfVectorizer(stop_words = 'english', max_features = 3000)
X = tfidf.fit_transform(df['text'])
y = df['label']

# Train-Test Split :
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)

# Training : (NBC)
start = time.time()
NBC = MultinomialNB(alpha = 1.0)
NBC.fit(X_train, y_train)
train_NBC = time.time() - start # Training Time

start = time.time() 
pred_nbc = NBC.predict(X_test) # Predictions
prob_nbc = NBC.predict_proba(X_test)[:, 1] # Probabilities
inf_nbc = (time.time() - start)/X_test.shape[0] # Inference Time

# Training : (BBC)
start = time.time()
BBC = BernoulliNB(alpha = 1.0)
BBC.fit(X_train, y_train)
train_BBC = time.time() - start

start = time.time()
pred_bbc = BBC.predict(X_test)
prob_bbc = BBC.predict_proba(X_test)[:, 1] 
inf_bbc = (time.time() - start)/X_test.shape[0] 

# Training : (Log-Reg)
start = time.time()
clf = LogisticRegression(max_iter = 1000)
clf.fit(X_train, y_train)
train_clf = time.time() - start

start = time.time()
pred_lr = clf.predict(X_test)
prob_lr = clf.predict_proba(X_test)[:, 1] 
inf_lr = (time.time() - start)/X_test.shape[0]


# Metrics : 
def evaluate(name, y_true, y_pred, y_prob = None):
    print(f"\n{name}")
    print("Classification Report : ", classification_report(y_true, y_pred))
    if y_prob is not None:
        print("ROC AUC:", roc_auc_score(y_true, y_prob))

evaluate("Multinomial NB", y_test, pred_nbc, prob_nbc)
evaluate("Bernoulli NB", y_test, pred_bbc)
evaluate("Logistic Regression", y_test, pred_lr, prob_lr)

# Confusion Matrix : 
cm = confusion_matrix(y_test, pred_nbc)
sns.heatmap(cm, annot = True, fmt = "d")
plt.title("Confusion Matrix :(Multinomial BC) ")
plt.show()

# ROC Comparison : 
fpr_nb, tpr_nb, _ = roc_curve(y_test, prob_nbc)
fpr_lr, tpr_lr, _ = roc_curve(y_test, prob_lr)

plt.plot(fpr_nb, tpr_nb, label = "Naive Bayes(Multinomial) : ")
plt.plot(fpr_lr, tpr_lr, label = "Logistic Regression : ")
plt.plot([0,1], [0,1], '--')
plt.legend()
plt.title("ROC Comparison : ")
plt.show()

# Top Spam Indicating Feature Names : 
feature_names = tfidf.get_feature_names_out()
spam_wts = NBC.feature_log_prob_[1]

top = np.argsort(spam_wts)[-20:]
plt.barh(feature_names[top], spam_wts[top])
plt.title("Top Spam Words : ")
plt.show()

# Training and Inference : 
print("\nTraining Time : ")
print("NB :", train_NBC)
print("LR :", train_clf)

print("\nInference Latency : ")
print("NB :", inf_nbc)
print("LR :", inf_lr)

