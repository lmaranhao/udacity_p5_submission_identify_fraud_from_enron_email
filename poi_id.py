#!/Users/leo/anaconda/bin/python

import sys
import pickle
#sys.path.append("/Users/leo/udacity_nanodegree/ud120-projects/tools/")

from feature_format import featureFormat, targetFeatureSplit, plot
from tester import dump_classifier_and_data
from pprint import pprint

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Not using the fields: email_address (because it's a string and doesn't add value to the model), and total_payments and total_stock_value (those are only total columns that sum up other features)
features_list = ['poi','salary', 'deferral_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

# print "features_list " + str(features_list)

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# plot function lines below are commented out
# they were used to check fo outliers 

#plot(data_dict, "salary", "bonus")
data_dict.pop('TOTAL', 0)
data_dict.pop('LOCKHART EUGENE E', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
pprint(data_dict)

### Task 3: Create new feature(s)
# here I created 2 new feature
# 1. I summed up the email from and to pois into the total_emails_related_to_poi feature
# 2. I calculated the percentage of the total emails that is related to pois and put it into the emails_percentage_related_to_poi feature 
for key in data_dict:
  if type(data_dict[key]["from_poi_to_this_person"]) is str and type(data_dict[key]["from_this_person_to_poi"]) is str:
    total_emails_related_to_poi = "NaN"
  elif type(data_dict[key]["from_poi_to_this_person"]) is str:
    total_emails_related_to_poi = data_dict[key]["from_this_person_to_poi"]
  elif type(data_dict[key]["from_this_person_to_poi"]) is str:
    total_emails_related_to_poi = data_dict[key]["from_poi_to_this_person"]
  else:
    total_emails_related_to_poi = data_dict[key]["from_poi_to_this_person"] + data_dict[key]["from_this_person_to_poi"]
  
  if type(data_dict[key]["from_messages"]) is str and type(data_dict[key]["to_messages"]) is str:
    total_emails = "NaN"
  elif type(data_dict[key]["from_messages"]) is str:
    total_emails = data_dict[key]["to_messages"]
  elif type(data_dict[key]["to_messages"]) is str:
    total_emails = data_dict[key]["from_messages"]
  else:
    total_emails = data_dict[key]["from_messages"] + data_dict[key]["to_messages"]

  if type(total_emails_related_to_poi) is str or type(total_emails) is str:
    emails_percentage_related_to_poi = "NaN"
  else:
    emails_percentage_related_to_poi = float(total_emails_related_to_poi) / float(total_emails)

  data_dict[key]["total_emails_related_to_poi"] = total_emails_related_to_poi
  data_dict[key]["emails_percentage_related_to_poi"] = emails_percentage_related_to_poi

### Store to my_dataset for easy export below.
my_dataset = data_dict

# ### PCA Goes here

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# automate pipeline and feature tuning (do not use svm)
# https://discussions.udacity.com/t/webcast-builidng-models-with-gridsearchcv-and-pipelines-thursday-11-feb-2015-at-6pm-pacific-time/47412

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors

pipeline = Pipeline([
        ('imputer', Imputer()),
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('clf', GaussianNB()),
    ])

param_grid = dict(
                  imputer__strategy = ["mean", "median"],
                  scaler__with_mean = [True, False],
                  scaler__with_std = [True, False],
                  scaler__copy = [True, False],
                  pca__n_components = [5, 10, 15, 'mle', None]
                  ) 

# cross-validation
validator = StratifiedShuffleSplit(n_splits = 10, test_size = 0.3, random_state = 42)

#create GridSearchCV object using param_grid
gridCV_object = GridSearchCV(estimator = pipeline, 
                                         param_grid = param_grid, 
                                         cv = validator)

#fit to the data
gridCV_object.fit(features_train, labels_train)

print "\nPipeline parameters:"
pprint(pipeline.get_params())

#what were the best parameters chosen from the parameter grid?
print "Best parameters from parameter grid:"
pprint(gridCV_object.best_params_)

#get the best estimator
clf = gridCV_object.best_estimator_

#check scores
print "\nPipeline Accuracy Score"
print clf.score(features_test, labels_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

