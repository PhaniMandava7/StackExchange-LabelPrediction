import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import fileinput

i=-1
testing_data=[]
data = []
with open("test_dataset.txt") as ft:
    for line in ft:
        data.append(line)

for line in data:
    if i==-1:
		no_test=int(line.split(" ")[0])
		i=i+1
    else:
        testing_data.append(json.loads(line))
        
testing_feature= [x['question'].lower() + " " + x['excerpt'].lower() for x in testing_data]

data = []
with open('training.json') as f:
    for line in f:
        data.append(json.loads(line))
training_feature =[x['question'].lower() + " " + x['excerpt'].lower() for x in data[1:]]
training_topics = [x['topic'] for x in data[1:]]


unique_topics =list(set(training_topics))
training_set = {}
training_inverse_set = {}
count=0
#print unique_topics
for x in unique_topics:
    count=count+1
    training_set[x]=count
    training_inverse_set[count] = x
#print training_set
#print training_inverse_set
training_class = [training_set[x] for x in training_topics]
#print training_class

vectorizer = TfidfVectorizer(stop_words='english', use_idf='True')
vectorized_feature = vectorizer.fit_transform(training_feature)
model = MultinomialNB().fit(vectorized_feature, training_class)

vectorized_feature_test = vectorizer.transform(testing_feature)

prediction = model.predict(vectorized_feature_test)
predicted_result=[]
for each_prediction in prediction:
    predicted_result.append(training_inverse_set[each_prediction])


actual = []
with open('actual_result.txt') as f:
    for line in f:
        actual.append(line)
count = 0
count1 = 0

for i in range(len(actual)):
    #print k[i],actual[i][:-1]
    if (str(predicted_result[i]) == str(actual[i][:-1])):
        count= count + 1
        print "Macth:" + str(count) + str(": ") + str(predicted_result[i])
    else:
		count1=count1+1
		print "Match not found:" + str(count1) + " : " + str(predicted_result[i]) + "\t" + "Actual:" + str(actual[i])

print "\n\n\n"
print "Total instances checked:" + str(len(predicted_result))
print "Total instances matches:" + str(count)
print "Total instances doesn't match:" + str(count1)
print "Accuracy of matching:"+ str(round((float(count*100))/len(actual),2)) + "%"
