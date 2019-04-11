import os
import io
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

train_texts = []
train_labels = []
test_texts = []
test_labels = []

labels = os.listdir(r'N:/text_classification/train')
#print(labels)

# 加载数据
for label in labels:
    train_path = 'N:/text_classification/train/' + label + '/'
    test_path = 'N:/text_classification/test/' + label + '/'
    for file in os.listdir(train_path):
        with open(train_path+file,encoding='gb18030',errors='ignore') as f:
            train_texts.append(f.readline())
            train_labels.append(label)

    for file in os.listdir(test_path):
        with open(test_path+file,encoding='gb18030',errors='ignore') as f:
            test_texts.append(f.readline())
            test_labels.append(label)
            

# 加载停用词
with open('N:/text_classification/stop/stopword.txt', 'rb') as f:
    STOP_WORDS = [line.strip() for line in f.readlines()]


# 使用jieba分词并计算单词权重
tf = TfidfVectorizer(tokenizer=jieba.cut, stop_words=STOP_WORDS, max_df=0.5)
train_features = tf.fit_transform(train_texts)
#print(train_features.shape)

# 生成朴素贝叶斯分类器
clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)

# 使用分类器做预测并计算准确率
test_tf = TfidfVectorizer(tokenizer=jieba.cut,stop_words=STOP_WORDS, max_df=0.5, vocabulary=tf.vocabulary_)
test_features=test_tf.fit_transform(test_texts)

#print(test_features.shape)
predicted_labels=clf.predict(test_features)
print('准确率为：',metrics.accuracy_score(test_labels, predicted_labels))
