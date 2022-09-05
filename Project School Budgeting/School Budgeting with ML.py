# Introducing the challenge

# 1. Budgets for schools are huge, complex, and not standardized
#	 - Hundreds of hours each year are spent manually labelling
# 2. Goal: Build a machine learning algorithm that can automate yhe process
# 3. Budget data
#	 - Line-item: 'Algebra books for 8th grade students'
#	 - Labels: 'Textbooks', 'Math', 'Middle School'
# 4. This is a supervised learning problem


# Exploring the data

#Load and preview the data#
import pandas as pd
samples_df = pd.read_csv('sample_data.csv')
sample_df.head()
sample_df.info()
samples_df.describe()

#Summarizing the data#

print(df.describe())
import matplotlib.pyplot as plt
plt.hist(df['FTE'].dropna())

plt.title('Distribution of %full-time \n employee works')
plt.xlabel('% of full-time')
plt.ylabel('num employees')
plt.show()

====================================================================
#Looking at the datatypes

#Objects instead of categories#
sample_df['label'].head()

#Encode labels as categories(sample data)#
sample_df.label.head(2)

sample_df.label = sample_df.label.astype('category')
sample_df.label.head(2)

#Dummy variable encoding#
dumies = pd.get_dummies(sample_df[['label']],prefix_sep='_')
dummies.head(2)

#Lambda functions#
square = lambda x: x*x
square(2)

#Encode labels as categories#
categories_label = lambda x: x.astype('category')
sample_df.label = sample_df[['label']].apply(categorize_label, axis=0)
sample_df.info()

#Encode the labels as categorical variables#

categorize_label = lambda x: x.astype('category')
df[LABELS] = df[LABELS].apply(categorize_label, axis=0)
print(df[LABELS].dtypes)


#Counting unique labels#

import matplotlib.pyplot as plt
num_unique_labels = df[LABELS].apply(pd.Series.nunique)
num_unique_labels.plot(kind='bar')

plt.xlabel('Labels')
plt.ylabel('Number of unique values')

plt.show()

==========================================================================
#How do we measure success?

#Computing log loss with NumPy#
import numpy as np
def comput_log_loss(predicted, actual, eps=le-14):
	""" Computes the logarithmic loss between predicted and 
	    actual when these are 10 arrays
	    
 	    :param predicted: The predicted probabilities as floats between 0-1
	    :param actual: The actual binary labels. Either 0 or 1.
	    :param eps (optional): log(0) is inf, so we need to offset our
			predicted values slightly by eps from 0 or 1
	"""
    predicted = np.clip(predicted, eps, 1-eps)
    loss = -1*np.mean(actual * np.log(predicted)
	     +(1 - actual)
	     *np.log(1 - predicted))
    return loss

compute_log_loss(predicted=0.9, actual=0)
compute_log_loss(predicted=0.5, actual=1)

#Computing log loss with NumPy#

correct_confident_loss = compute_log_loss(correct_confident, actual_labels)
print("Log loss, correct and confident: {}".format(correct_confident_loss)) 

correct_not_confident_loss = compute_log_loss(correct_not_confident, actual_labels)
print("Log loss, correct and not confident: {}".format(correct_not_confident_loss)) 

wrong_not_confident_loss = compute_log_loss(wrong_not_confident, actual_labels)
print("Log loss, wrong and not confident: {}".format(wrong_not_confident_loss)) 

wrong_confident_loss = compute_log_loss(wrong_confident, actual_labels)
print("Log loss, wrong and confident: {}".format(wrong_confident_loss)) 

actual_labels_loss = compute_log_loss(actual_labels, actual_labels)
print("Log loss, actual labels: {}".format(actual_labels_loss)) 

===============================================================
#It's time to build a model

#Spliting the multi-class dataset#
- Solution: StratifiedshuffleSplit
	-multilabel_train_test_split()

#Splitting the data#

data_to_train = df[NUMERIC_COLUMNS].fillna(-1000)
labels_to_use = pd.get_dummies(df[LABELS])
X_train, X_test, y_train, y_test = multilabel_train_test_split(
			data_to_train,
			labels_to_use,
			size=0.2, seed=123)

#Training the model#
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)


#Setting up a train-test split in scikit-learn#

numeric_data_only = df[NUMERIC_COLUMNS].fillna(-1000)
label_dummies = pd.get_dummies(df[LABELS])

X_train, X_test, y_train, y_test = multilabel_train_test_split(      
                                                    numeric_data_only,
                                                    label_dummies,
                                                    size=0.2, 
                                                    seed=123)

# Print the info
print("X_train info:")
print(X_train.info())
print("\nX_test info:")  
print(X_test.info())
print("\ny_train info:")  
print(y_train.info())
print("\ny_test info:")  
print(y_test.info()) 


#Training a model#

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

numeric_data_only = df[NUMERIC_COLUMNS].fillna(-1000)
label_dummies = pd.get_dummies(df[LABELS])

X_train, X_test, y_train, y_test = multilabel_train_test_split(numeric_data_only,
                                                               label_dummies,
                                                               size=0.2, 
                                                               seed=123)

clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)
print("Accuracy: {}".format(clf.score(X_test, y_test)))

======================================================================
#Making predictions

#Predicting in holdout data#
holdout = pd.read_csv('HoldoutData.csv', index_col=0)
holdout = holdout[NUMERIC_COLUMNS].fillna(-1000)
predictions = clf.predict_proba(holdout)

#Format and submit predictions#

prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS],
			     prefix_sep='__').columns,
			     index=holdout.index,
			     data=predictions)
prediction_df.to_csv('predictions.csv')
score = score_submission(pred_path='predictions.csv')

=====================================================================
#A very brief introduction to NLP
#Representing text numerically

#Using CountVectorizer() on column of main dataset#

from sklearn.feature_extraction.text import CountVectorizer
TOKENS_BASIC = '\\\\S+(?=\\\\s+)'
df.Program_Description.fillna('', inplace=True)
vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC)

vec_basic.fit(df.Program_Description)
msg = 'There are {} tokens in Program_Description if tokens are any non-whitespace'
print(msg.format(len(vec_basic.get_fecture_names())))

#What's in a token?#

from sklearn.feature_extraction.text import CountVectorizer
TOKENS_BASIC = '\\S+(?=\\s+)'
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC)
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

text_vector = combine_text_columns(df)
vec_basic.fit_transform(text_vector)
print("There are {} tokens in the dataset".format(len(vec_basic.get_feature_names())))
vec_alphanumeric.fit_transform(text_vector)
print("There are {} alpha-numeric tokens in the dataset".format(len(vec_alphanumeric.get_feature_names())))

=======================================================================================
#Pipelines, feature & text preprocessing

#Instantiate simple pipeline with one step#
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.muticlass import OneVsRestClassifier
pl = pipeline([
	('clf', OneVsRestClassifier(LogisticRegression()))
    ])

#Train and test with sample numeric data#
sample_df.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
					sample_df[['numeric']],
	 				pd.get_dummies(sample_df['label']),
					random_state=2)
pl.fit(X_train, y_train)
accuracy = pl.score(X_test, y_test)
print('accuracy on numeric data, no nans: ', accuracy)


#Adding more steps to the pipeline#

X_train, X_test, y_train, y_test = train_test_split(
					sample_df[['numeric','with_missing']],
	 				pd.get_dummies(sample_df['label']),
					random_state=2)
pl.fit(X_train, y_train)


#Preprocessing numeric features with missing data#

from sklearn.preprocessing import Imputer
X_train, X_test, y_train, y_test = train_test_split(
					sample_df[['numeric',with_missing]],
	 				pd.get_dummies(sample_df['label']),
					random_state=2)
pl = Pipeline([
	('imp', Imputer()),
	('clf', OneVsRestClassifier(LogisticRegression()))
    ])

pipeline.fit(X_train, y_train)
accuracy = pl.score(X_test, y_test)
print('accuracy on numeric data, no nans: ', accuracy)

========================================================================
#Text features and feature unions

#Preprocessing text features#

from sklearn.feature_extraction.text import CountVectorizer
X_train, X_test, y_train, y_test = train_test_split(
					sample_df['text'],
	 				pd.get_dummies(sample_df['label']),
					random_state=2)
pl = Pipeline([
	('vec', CountVectorizer())
	('clf', OneVsRestClassifier(LogisticRegression()))
  ])

#Preprocessing text features#

pl.fit(X_train, y_train)
accuracy = pl.score(X_test, y_test)
print('accuracy on sample data: ', accuracy)

#Putting it all together#
X_train, X_test, y_train, y_test = train_test_split(
					sample_df[['numeric', 'with_missing', 'text']],
	 				pd.get_dummies(sample_df['label']),
					random_state=2)
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion

get_text_data = FunctionTransformer(lamvda x: x['text'],validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[['numeric', 'with_missing']], validate=False)


#FeatureUnion Text and Numeric Features#

from sklearn.pipeline import FeatureUnion
union = FeatureUnion([
		('numeric', numeric_pipeline),
		('text', text_pipeline)	
        ])

#Putting it all together#

numeric_pipeline = Pipeline([
			('selector', get_numeric_data),
			('imputer', Imputer())	
		    ])
text_pipeline = Pipeline([
			('selector' , get_text_data),	
			('vectorizer', CountVectorizer())	
		])
pl = Pipeline([
	      ('union', FeatureUnion([
	      ('numeric', numeric_pipeline),
	      ('text' , text_pipeline)
	])),
	('clf' , OneVsRestClassifier(LogisticRegression()))
	])

=================================================================
#Choosing a classification model

#Main dataset: lots of text#
LABELS = ['Function', 'Use', 'Sharing', 'Reporting', 'Student_Type',
	  'Possition_Type', 'Object_Type', 'Pre_K', 'Operating_Status']
NON_LABELS = [c for c in df.columns if c not in LABELS]
len(NON_LABELS) - len(NUMERIC_COLUMNS)

#Using pipeline with the main dataset#

import numpy as np
import pandas as pd
df = pd.read_csv('TrainingSetSample.csv', index_col=0)
dummy_labels = pd.get_dummies(df[LABELS])
X_train, X_test, y_train, y_test = multilabel_train_test_split(
				   df[NON_LABELS], dummy_labels,
				   0.2)

get_text_data = FunctionTransformer(combine_text_columns,
					validate=False)
get_numeric_data = FunctionTransformer(lambda x:
			x[NUMERIC_COLUMNS],validate=False)
pl = Pipeline([
	     ('union', FeatureUnion([
			('numeric_features', Pipeline([
				('selector', get_numeric_data),
			]))
			('text_featurees', Pipeline([
				('selector', get-text_data),
				('vectorizer', Countvectorizer())
			]))
		    ])
		),
		('clf', OneVsRestClassifier(LogisticRegression()))
	     ])


#Performance using main dataset#
pl.fit(X_train, y_train)


#Easily try new models using pipeline#

from sklearn.ensemble import RandomForestClassifier
pl = Pipeline([
	     ('union', FeatureUnion(
		   transformer_list = [
			('numeric_features', Pipeline([
				('selector', get_numeric_data),
				('imputer' , Umputer())
			])),
			('text_featurees', Pipeline([
				('selector', get-text_data),
				('vectorizer', Countvectorizer())
			]))
		    ]
 	     )),
		('clf', OneVsRestClassifier(RandomForestClassifier()))
	     ])

===========================================================================
#Learning from the expert: processing

#N-grams and tokenization#
vec = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
		      ngram_range=(1,2))

#Range of n-grams in scikit-learn#
pl.fit(X_train, y_train)

holdout = pd.read_csv('HoldoutData.csv', index_col=0)
predictions = pl.predict_preba(holdout)
prediction_df = pd.DataFrame(columns=pd.get_dummies(
		df[LABELS]).columns, index=holdout.index,
		data=predictions)
prediction_df.to_csv('predictions.csv')
score = score_submission(pred_path='predictions.csv')


===========================================================================
#Learning from the expert: a stats trick

#Adding interaction features with scikit-learn#

from sklearn.preprocessing import PolynomialFeatures
x

interaction = PolynomialFeatures(degree=2,
				 interaction_only=True,
				 include_bias=False)
interaction.fit_transform(x)


#Sparse interaction features#
SparseInteractions(degree=2).fit_transform(x).toarray()

===========================================================================
#Learning from the expert: the winning model

#Implementing the hashing trick in scilit-learn#
from sklearn.feature_extraction.text import HashingVectorizer

vec = HashingVectorizer(norm=None,
			non_negative=True,
			token_pattern=TOKENS_ALPHANUMERIC,
			ngram_range=(1,2))


#Implementing the hashing trick in scikit-learn#


from sklearn.feature_extraction.text import HashingVectorizer

text_data = combine_text_columns(X_train)
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)' 

hashing_vec = HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC)
hashed_text = hashing_vec.fit_transform(text_data)
hashed_df = pd.DataFrame(hashed_text.data)
print(hashed_df.head())
