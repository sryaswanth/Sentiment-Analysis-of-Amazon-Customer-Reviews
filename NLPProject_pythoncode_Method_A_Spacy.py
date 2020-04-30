# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:45:32 2020

@author: sryas
"""

import pandas as pd
import spacy
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


from sklearn.neural_network import MLPClassifier


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report,confusion_matrix

import keras as Ker

from keras.models import Sequential

from keras.layers import Dense

import math



nlpPreprocess=spacy.load('en_core_web_sm')
data=pd.read_csv("amazon_reviews_us_Mobile_Electronics_v1_00.tsv", sep='\t',error_bad_lines=False)
data=data.drop(['marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent', 'helpful_votes', 'total_votes', 'vine'], axis = 1) 


#loading sentinet file
sentinet_data = pd.read_excel("sentinet.xlsx")

#____________________________________
#to get the list of stop words in spacy

#spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
#print("Spact stop words list: \n", list(spacy_stopwords))
#print("Spacy stop words count: ", len(spacy_stopwords))


#____________________________________

#spacy preprocessing - stopwords removal and lemmatization

def spacy_preprocessing_function(data):
    print(data["review_headline"].tail(10))
    print(data["review_body"].tail(10))

    datasetLen=len(data)

    for eachCell in range(0,datasetLen):
        eachCellListH=[]
        eachCellListB=[]
        eachCellContentH=data["review_headline"].get(eachCell)
        eachCellContentB=data["review_body"].get(eachCell)
        for tokenH,tokenB in zip(nlpPreprocess(str(eachCellContentH)),nlpPreprocess(str(eachCellContentB))):
            if tokenH.is_alpha and not tokenH.is_stop:
                eachCellListH.append(str(tokenH.lemma_).lower())
            if tokenB.is_alpha and not tokenB.is_stop:
                eachCellListB.append(str(tokenB.lemma_).lower())
        data["review_headline"][eachCell]=eachCellListH
        data["review_body"][eachCell]=eachCellListB

    print(data["review_headline"].tail(10))
    print(data["review_body"].tail(10))
    
    return data


#____________________________________________________________________________________
#if data loaded from external CSV file, convert the columns to list


def preprocessing_headline_and_body_after_importing_data(data):

    data["review_headline"] = data["review_headline"].str.strip('[]')

    data["review_headline"] = data["review_headline"].replace({"'":""}, regex = True)

    data["review_headline"] = data["review_headline"].replace({" ":""}, regex = True)

    data["review_headline"] = data["review_headline"].str.split(",")

    

    data["review_body"] = data["review_body"].str.strip('[]')

    data["review_body"] = data["review_body"].replace({"'":""}, regex = True)

    data["review_body"] = data["review_body"].replace({" ":""}, regex = True)

    data["review_body"] = data["review_body"].str.split(",")

    return data
#_________________________________________________________________________________________

#getting the words in each review to compare with sentinet file for REVIEW HEADLINE

def function_to_comare_words_in_dataset_with_sentinet_file_for_review_headline(data,sentinet_data):
    word_count = []
    intensity_list = []
    for i in data["review_headline"]:
        total_words = 0;
        intensity =0;
        for j in i:
        
            temporary_intensity = 0;
            index_value = 0;
        
            for k in sentinet_data["CONCEPT"]:
            
                if(k == j):
                    index = index_value
                    temporary_intensity = sentinet_data.get_value(index,'INTENSITY')
                    intensity = intensity +  temporary_intensity 
                      
                index_value+=1
            total_words+=1     
        word_count.append(total_words)
        intensity_list.append(intensity)
        
        print(word_count)
        print(intensity)
        
    print(word_count)
    print(intensity_list)

    word_count_df = pd.DataFrame(word_count)
    word_count_df.to_csv('word_count_headline.csv',index=False, header=False)

    intensity_list_df = pd.DataFrame(intensity_list)
    intensity_list_df.to_csv('intensity_headline.csv', index=False, header=False)
    
    return intensity_list
#___________________________________________________________________________________

#getting the words in each review to compare with sentinet file for REVIEW body
def function_to_comare_words_in_dataset_with_sentinet_file_for_review_body(data,sentinet_data):
    word_count_body = []
    intensity_list_body = []
    for i in data["review_body"]:
        total_words = 0;
        intensity =0;
        for j in i:
        
            temporary_intensity = 0;
            index_value = 0;
        
            for k in sentinet_data["CONCEPT"]:
            
                if(k == j):
                    index = index_value
                    temporary_intensity = sentinet_data.get_value(index,'INTENSITY')
                    intensity = intensity +  temporary_intensity 
                      
                index_value+=1
            total_words+=1     
        word_count_body.append(total_words)
        intensity_list_body.append(intensity)
        
        print(word_count_body)
        print(intensity_list_body)
    
    print(word_count_body)
    print(intensity_list_body)

    word_count_df_body = pd.DataFrame(word_count_body)
    word_count_df_body.to_csv('word_count_body.csv',index=False, header=False)

    intensity_list_df_body = pd.DataFrame(intensity_list_body)
    intensity_list_df_body.to_csv('intensity_body.csv', index=False, header=False)

    return intensity_list_body



#______________________________________________________________________________________
#Create classes for the dataset (headline and body)
    
#using for prediefined dataset
    
def create_classes_P_N_N_for_the_data(data_with_intensity):
    
    body_class = []
    headline_class = []
    interation_no = 0;
    
    positive_count = 0
    negative_count = 0
    neutral_count =  0


    for i in data_with_intensity["Body_intensity"]:
        
        String = ""
        if(i > 0):
            String = "Positive"
            positive_count+=1
        elif(i<0):
            String = "Negative"
            negative_count+=1
        elif(i == 0):
            if(math.isnan(data_with_intensity.at[interation_no,"star_rating"])):
                String = "Negative"
                negative_count+=1
            elif(data_with_intensity.at[interation_no,"star_rating"] >= 3):
                String = "Positive"
                positive_count+=1
            elif(data_with_intensity.at[interation_no,"star_rating"] < 3):
                String = "Negative"
                negative_count+=1
                
                
       #     neutral_count+=1
        
        interation_no = interation_no + 1;
        body_class.append(String)

    data_with_intensity["Review_Body_CLASS"] = body_class
    
    x = np.array(body_class)
    print("class creation")
    print(data_with_intensity["star_rating"].value_counts('nan'))
    
    print(np.unique(x)) 
      


    for i in data_with_intensity["Headline_intensity"]:
        String = ""
        if(i > 0):
            String = "Positive"
        elif(i<0):
            String = "Negative"
        elif(i == 0):
            String = "Neutral"
            
    
        headline_class.append(String)

    data_with_intensity["Review_Headline_CLASS"] = headline_class

    return data_with_intensity, body_class, headline_class


#___________________________________________________________________________________________
    
def create_classes_P_N_N_for_the_data_manual(data_with_intensity):
    
    body_class = []
    
    positive_count = 0
    negative_count = 0


    for i in data_with_intensity["Body_intensity"]:
        
        String = ""
        if(i > 0):
            String = "Positive"
            positive_count+=1
        elif(i <= 0):
            String = "Negative"
            negative_count+=1
        
        body_class.append(String)
    
    data_with_intensity["Review_Body_CLASS"] = body_class
    return data_with_intensity, body_class

    


#______________________________________________________________________________________

#convertlist to string for vectorization
    
#for body cvolumn

def convert_list_to_string_for_vectorization_body(data_with_intensity):
    
    #for review_body
    body_as_string = []

    for i in data_with_intensity["review_body"]:
        String = ""
        String = ' '.join(i)
        body_as_string.append(String)
        
    return body_as_string

#for headline column
    

def convert_list_to_string_for_vectorization_headline(data_with_intensity):

    #for review_body
    headline_as_string = []
    for i in data_with_intensity["review_headline"]:
        String = ""
        String = ' '.join(i)
        headline_as_string.append(String)
        
    return headline_as_string


#_________________________________________________________________________________________
  
#Naivebayes function

def perform_NaiveBayes_algorithm(train_data, test_data, train, test, stratified_data):
    
    
    Naivebayes_algorithm = MultinomialNB()
    NB = Naivebayes_algorithm.fit(train_data,train['body_class'])
    
    #predicting the classes for the test data
    prediction = NB.predict(test_data)

    print("Accuracy score without corss validation:",accuracy_score(test['body_class'],prediction))
    print("F1 Score without cross validation:", f1_score(test['body_class'],prediction, average = 'micro'))
    
    print("Confusion Matrix",confusion_matrix(test['body_class'],prediction))
    
    return prediction

#___________________________________________________________________________
#Random forest function

def perform_Randomforest_algorithm(train_data, test_data, train, test, stratified_data):
    
    random_forest = RandomForestClassifier(max_depth=100, random_state=0)

    random_forest.fit(train_data,train['body_class'])

    prediction = random_forest.predict(test_data)
    print("Accuracy score_ RF without cross validation:",accuracy_score(test['body_class'],prediction))
    print("F1 Score without cross validation:", f1_score(test['body_class'],prediction, average = 'micro'))
    
    print("Confusion Matrix",confusion_matrix(test['body_class'],prediction))
    
    return prediction

#____________________________________________________________________________________
    
#SVM algorithm function
    

def perform_SVM_algorithm(train_data, test_data, train, test, stratified_data):
    
    SVM_algorithm = SVC(gamma = 'scale')

    SVM_algorithm.fit(train_data,train['body_class'])
    
    prediction = SVM_algorithm.predict(test_data)
    print("Accuracy score_ SVM:",accuracy_score(test['body_class'],prediction))
    print("F1 Score without cross validation:", f1_score(test['body_class'],prediction, average = 'micro'))

    print("Confusion Matrix",confusion_matrix(test['body_class'],prediction))
    
    return prediction


#________________________________________________________________________________________
    
#neural network function

def perform_Neural_netwrok_function(train_data, test_data, train, test, stratified_data):


    Neural_net = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 15), random_state=1)

   # Neural_net = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


    Neural_net.fit(train_data,train['body_class'])

    prediction = Neural_net.predict(test_data)

    print("Accuracy (Neural network): ",accuracy_score(test['body_class'],prediction))
    print("F1 Score without cross validation(Neural Network):", f1_score(test['body_class'],prediction, average = 'micro'))


    print("Confusion Matrix",confusion_matrix(test['body_class'],prediction))

    print("Classification Report:",classification_report(test['body_class'],prediction))


    print("Weights between each layers:", Neural_net.coefs_)

    return prediction
#_____________________________________________________________________________________
    
def confusion_matrix_function(prediction, test_y):
    
    #but what if the dataset is skewed(yes, our data is skewed!)
    #lets try confusion matrix to find out true positives and negatives

    conf_matrix = confusion_matrix(y_true = test_y, y_pred = prediction)

    labels = ['Class 0', 'Class 1', 'class 2']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted result')
    plt.ylabel('Expected result')
    plt.show()
    


#____________________________________________________________________________________
    

#function to perform undersmapling (just to see if we get some accuracy improvements)


#for body column
def undersampling_function_for_body_cloumn(data_with_intensity):
    
    #lets try undersampling

    #counts of each class

    negative_count = len(data_with_intensity[data_with_intensity['Review_Body_CLASS'] == 'Negative'])
    print(negative_count)


    #finding the indexes of positive classes
    positive_indexes = data_with_intensity[data_with_intensity.Review_Body_CLASS == 'Positive'].index
    print(positive_indexes)

    #finding the indexes of Neutral classes
    neutral_indexes = data_with_intensity[data_with_intensity.Review_Body_CLASS == 'Neutral'].index
    print(neutral_indexes)

    # rand sampling of positive class indices
    random_indices_positive = np.random.choice(positive_indexes,negative_count, replace= False)

    # rand sampling of neutral class indices
    random_indices_neutral  = np.random.choice(neutral_indexes,negative_count, replace= False)


    #finding the indexes of negative classes
    negative_indexes = data_with_intensity[data_with_intensity.Review_Body_CLASS == 'Negative'].index
    print(negative_indexes)

    under_sampling_indexes = np.concatenate([negative_indexes,random_indices_positive,random_indices_neutral])
    
    undersampled_data = data_with_intensity.loc[under_sampling_indexes]
    
    return undersampled_data

#___________________________________________________________________________________






#_____________________________________________________________________________________
#only run this code when we want to do spacy preprocessing again

#data = spacy_preprocessing_function(data)

#export preprocessed data

#data.to_csv(r'D://preprocessed_data.csv', index = False)

#load exported data

#data = pd.read_csv("preprocessed_data.csv")



#_________________________________________________________________________________________


#use this piece of code only when you want to preprocess again (to compare words with sentinet)

#appending Review headline and body intensities to the data frame


#data = preprocessing_headline_and_body_after_importing_data(data)

#data_with_intensity = pd.DataFrame(data)

#intensity_list = function_to_comare_words_in_dataset_with_sentinet_file_for_review_body(data,sentinet_data)
#intensity_list_body = function_to_comare_words_in_dataset_with_sentinet_file_for_review_headline(data,sentinet_data)

#data_with_intensity["Headline_intensity"] = intensity_list
#data_with_intensity["Body_intensity"] = intensity_list_body

#export dataframe as csv
#data_with_intensity.to_csv(r'D://data_with_intensity.csv', index = False)

#_______________________________________________________________________________________


# this is where we need to start running (dont run above code except for the functions and import data)

#load the data_with_intensity_file

data_with_intensity = pd.read_csv("data_with_intensity.csv")

#if data loaded from external CSV file, convert the columns to list
data_with_intensity = preprocessing_headline_and_body_after_importing_data(data_with_intensity)

print(data_with_intensity["review_headline"])
print(data_with_intensity["review_body"])


#__________________________________________________________________________________________

#creating classes (positive and negative) for headline and body

data_with_intensity, body_class,headline_class  = create_classes_P_N_N_for_the_data(data_with_intensity)

print(data_with_intensity)


#________________________________________________________________________________________-

#converting list to string in Dataframe body and headline columns for vectorization

body_as_string = convert_list_to_string_for_vectorization_body(data_with_intensity)

headline_as_string = convert_list_to_string_for_vectorization_headline(data_with_intensity)


#__________________________________________________________________________________________

#creating dataframe to stratify the dataset

stratified_data = pd.DataFrame()

stratified_data['body_as_string'] = body_as_string

stratified_data['body_class'] = body_class



#______________________________________________________________________________________

#split the data into training and testing

train, test = train_test_split(stratified_data, test_size = 0.2,stratify = stratified_data['body_class'] )

print(train.groupby(['body_class']).count())
print(test.groupby(['body_class']).count())

#____________________________________________________________________________________-
#vectorizing the data

count_vectorizer = CountVectorizer()
Tfidf_vectorizer = TfidfVectorizer()

count_vectorizer.fit(train['body_as_string'])

training_data_count = count_vectorizer.transform(train['body_as_string'])


Tfidf_vectorizer.fit(train['body_as_string'])

training_data_tfidf = Tfidf_vectorizer.transform(train['body_as_string'])

# to find the vocabular size after spacy preprocessing
#count_vectorizer.fit(stratified_data['body_as_string'])

#for_count = count_vectorizer.transform(stratified_data['body_as_string'])

#print(for_count.shape)


#_____________________________________________________________


#naivebayes function

#for count_vectorizer
#prediction = perform_NaiveBayes_algorithm(training_data_count, testing_data_count,train,test, stratified_data)


#for TF-IDFvectorizer
#prediction = perform_NaiveBayes_algorithm(training_data_tfidf, testing_data_tfidf,train,test, stratified_data)



#____________________________________________________________________-

#function for confusion matrix (try only if you want to)
#confusion_matrix_function(prediction, test['body_class'])


#____________________________________________________________________


#function to perform undersmapling (just to see if we get some accuracy improvements)

#undersampled_data_body = data_with_intensity

#undersampled_data_body = undersampling_function_for_body_cloumn(undersampled_data_body)

#body_as_string = convert_list_to_string_for_vectorization_body(undersampled_data_body)

#train_X_count_body = count_vectorizer.fit_transform(body_as_string)

#train_X_Tfidf_body = Tfidf_vectorizer.fit_transform(body_as_string)


#training and predictiong the data for undersampled data

#naivebayes function

#for count_vectorizer
#prediction = perform_NaiveBayes_algorithm(train_X_count_body, undersampled_data_body['Review_Body_CLASS'])

#for TF-IDFvectorizer
#prediction = perform_NaiveBayes_algorithm(train_X_Tfidf_body, undersampled_data_body['Review_Body_CLASS'])


#_______________________________________________________________________________________

#applying random forest algorithm

#not if you perform undersampling before this please adjust the body_class again, since mostly the count wont match

#for count_vectorizer
#prediction = perform_Randomforest_algorithm(training_data_count, testing_data_count,train,test, stratified_data)

#for TF-IDFvectorizer
#prediction = perform_Randomforest_algorithm(training_data_count, testing_data_count,train,test, stratified_data)


#____________________________________________________________________________________

#with SVM algorithm

#not if you perform undersampling before this please adjust the body_class again, since mostly the count wont match

#for count_vectorizer
#prediction = perform_SVM_algorithm(training_data_count, testing_data_count,train,test, stratified_data)

#for TF-IDFvectorizer
#prediction = perform_SVM_algorithm(training_data_count, testing_data_count,train,test, stratified_data)


#______________________________________________________________________________________________

# Neural network


#for count_vectorizer
#prediction = perform_Neural_netwrok_function(training_data_count, testing_data_count,train,test, stratified_data)

#for TF-IDFvectorizer
#prediction = perform_Neural_netwrok_function(training_data_count, testing_data_count,train,test, stratified_data)

#__________________________________________________________________________________________

def ANN_function(train, test, training_data_count,shape):
    
    #here 'train' is the teaing data after train test split
    #here 'test' is the testing data after train test split
    # here "training_data_count" is the vectorized training data

    df = pd.DataFrame(train['body_class']) 

    df = df.replace({'Positive' : 1 , 'Negative' : 0} )



    #intializing the ANN

    ANN = Sequential()



    #creating the input layer 

    ANN.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu', input_dim = shape))

    #adding the first hidden layer

    ANN.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
    
      #adding the second hidden layer
    
    ANN.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
    
      #adding the third hidden layer
    
    ANN.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
    
     #adding the fourth hidden layer
    
    ANN.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
     

    #addign the output layer
    ANN.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

    #now compile the ANN

    ANN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


    #fitting the ANN to the training set
    
    ANN.fit(training_data_count,df, batch_size = 10, nb_epoch = 20)

    #  y_pred = ANN.predict(testing_data_count)
#    print(accuracy_score(df_test,y_pred))

    #default testing dataset
    testing_data_count = count_vectorizer.transform(test['body_as_string'])
    
    df_test = pd.DataFrame(test['body_class']) 

    df_test = df_test.replace({'Positive' : 1 , 'Negative' : 0} )
    
    y_pred = ANN.predict(testing_data_count)
    
    new_y_pred = []
    
    for item in y_pred:
        if(item > 0.5):
            new_y_pred.append(1)
        elif(item<=0.5):
            new_y_pred.append(0)
            
    
    
    print(accuracy_score(df_test,new_y_pred))
    
    return new_y_pred


    # own sentence

 #   manual_data = pd.DataFrame(columns = ["review_body","review_headline"])
  #  input_sentence = input("Enter a review(sentence) as input:")

   # manual_data.set_value(0,'review_body',input_sentence)
    

  #  manual_data = spacy_preprocessing_function(manual_data)

   # intensity_list = function_to_comare_words_in_dataset_with_sentinet_file_for_review_body(manual_data,sentinet_data)



 #   manual_data['Body_intensity'] = intensity_list
 #   manual_data['Headline_intensity'] = 0
 #   manual_data, body_class, headline_class = create_classes_P_N_N_for_the_data(manual_data)

  #  body_as_string = convert_list_to_string_for_vectorization_body(manual_data)

 #   stratified_data = pd.DataFrame()
    
#    stratified_data['body_as_string'] = body_as_string

#    stratified_data['body_class'] = body_class


#    testing_data_count = count_vectorizer.transform(stratified_data['body_as_string'])



  #  y_pred = ANN.predict(testing_data_count)
#    print(accuracy_score(df_test,y_pred))

 #   cm = confusion_matrix(y_pred,df_test)




#__________________________________________________________________________________


#ANN calling function

print(shape[1])



#__________________________________________________________________________________

#decide what to do


decision = input("Do you want to get the accuracy results for the default test set (please choose 'y/n'):")

if(decision == 'y'):
    
    testing_data_count = count_vectorizer.transform(test['body_as_string'])
    testing_data_tfidf = Tfidf_vectorizer.transform(test['body_as_string'])
    
    while True:
        algo = input('which algorithm would you like to run (choose one among(NB, SVM, RF, MLP_NN, ANN)):')
        if(algo == 'NB'):
            #naivebayes function

            #for count_vectorizer
            prediction = perform_NaiveBayes_algorithm(training_data_count, testing_data_count,train,test, stratified_data)

            #for TF-IDFvectorizer
            prediction = perform_NaiveBayes_algorithm(training_data_tfidf, testing_data_tfidf,train,test, stratified_data)
    
        if(algo == 'SVM'):
                
            #for count_vectorizer
            prediction = perform_SVM_algorithm(training_data_count, testing_data_count,train,test, stratified_data)

            #for TF-IDFvectorizer
            prediction = perform_SVM_algorithm(training_data_tfidf, testing_data_tfidf,train,test, stratified_data)

        if(algo == 'RF'):

            #for count_vectorizer
            prediction = perform_Randomforest_algorithm(training_data_count, testing_data_count,train,test, stratified_data)
        
            #for TF-IDFvectorizer
            prediction = perform_Randomforest_algorithm(training_data_count, testing_data_count,train,test, stratified_data)

        if(algo == 'MLP_NN'):
  
            #for count_vectorizer
            prediction = perform_Neural_netwrok_function(training_data_count, testing_data_count,train,test, stratified_data)

            #for TF-IDFvectorizer
            prediction = perform_Neural_netwrok_function(training_data_count, testing_data_count,train,test, stratified_data)
        
        
        if(algo == 'ANN'):
            
            shape = training_data_count.shape
            
            #for count_vectorizer
            ypred = ANN_function(train, test, training_data_count,shape[1])

        
        
    
        
        run_again = input("Would you like to test again for another algorithm:(y/n):")
        if(run_again != 'y'):
            break
    

#___________________________________________________________________________________

#test with input data

manual_data = pd.DataFrame(columns = ["review_body","review_headline"])

while True:

    input_sentence = input("Enter a review(sentence) as input:")

    manual_data.set_value(0,'review_body',input_sentence)


    manual_data = spacy_preprocessing_function(manual_data)


    intensity_list = function_to_comare_words_in_dataset_with_sentinet_file_for_review_body(manual_data,sentinet_data)



    manual_data['Body_intensity'] = intensity_list

    manual_data['Headline_intensity'] = 0

    manual_data, body_class = create_classes_P_N_N_for_the_data_manual(manual_data)

    body_as_string = convert_list_to_string_for_vectorization_body(manual_data)

    stratified_data = pd.DataFrame()

    stratified_data['body_as_string'] = body_as_string

    stratified_data['body_class'] = body_class


    testing_data_count = count_vectorizer.transform(stratified_data['body_as_string'])

    testing_data_tfidf = Tfidf_vectorizer.transform(stratified_data['body_as_string'])


    while True:
            algo = input('which algorithm would you like to run (choose one among(NB, SVM, RF, NN)):')
            if(algo == 'NB'):
                #naivebayes function

                #for count_vectorizer
                prediction = perform_NaiveBayes_algorithm(training_data_count, testing_data_count,train,stratified_data, stratified_data)

                #for TF-IDFvectorizer
                prediction = perform_NaiveBayes_algorithm(training_data_tfidf, testing_data_tfidf,train,stratified_data, stratified_data)
    
            if(algo == 'SVM'):
                
                #for count_vectorizer
                prediction = perform_SVM_algorithm(training_data_count, testing_data_count,train,stratified_data, stratified_data)

                #for TF-IDFvectorizer
                prediction = perform_SVM_algorithm(training_data_count, testing_data_count,train,stratified_data, stratified_data)

            if(algo == 'RF'):

                #for count_vectorizer
                prediction = perform_Randomforest_algorithm(training_data_count, testing_data_count,train,stratified_data, stratified_data)
        
                #for TF-IDFvectorizer
                prediction = perform_Randomforest_algorithm(training_data_tfidf, testing_data_tfidf,train,stratified_data, stratified_data)
            
            if(algo == 'NN'):
  
                #for count_vectorizer
                prediction = perform_Neural_netwrok_function(training_data_count, testing_data_count,train,stratified_data, stratified_data)

                #for TF-IDFvectorizer
                prediction = perform_Neural_netwrok_function(training_data_tfidf, testing_data_tfidf,train,stratified_data, stratified_data)
            
            run_again = input("Would you like to test again for another algorithm:(y/n):")
            if(run_again != 'y'):
                break

    continue_manual_check = input("would you like to continue with another review? (y/n):")
    if(continue_manual_check!= 'y'):
        break



