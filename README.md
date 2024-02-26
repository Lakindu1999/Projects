# Predictive Model Building for Student Achievement in Secondary Education of Two Portuguese Schools

## 1. Introduction
The objective of this project is to analyze and model the factors influencing student achievement in secondary education, specifically focusing on the subject of Portuguese. The dataset at hand comprises information on student grades, along with demographic, social, and school-related features. the goal is to build a predictive model for the target variable G2.Port. Moreover the target variable G2.Port is to be binned into four categories, ensuring roughly equal distribution of cases across these bins. The resulting categorical variable will be used as the response for a classification model. The model should be constructed without utilizing any of the other grade features, ensuring a comprehensive exploration of non-grade related factors. For this, python is used for data analysis and building models using several packages like pandas, NumPy, Matplotlib, Seaborn and scikit-learn.

## 2. Background of the Data
The data in the file is about student achievement in secondary education of two Portuguese schools. The data attributes include student grades, demographic, social and school related features and it was collected by using school reports and questionnaires.
Here comes the explanation of the features included in the data set:

For the first step of the work, we have to import our main libraries, after that lets quick lookup of our dataset.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/2c1b87ba-a429-4030-b034-d1605f6fff04)

## 3. Data Preprocessing
We will also check if we need some data cleaning like removing null or missing values, dropping unnecessary columns and also if datatype of any variable is required to be changed. So, it will be done in the way described below.
Now we can check the basic summary of the dataset.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/bb978784-6652-4fb7-875f-5a80a7cde50c)

Discriptive analysis of our dataset
![image](https://github.com/Lakindu1999/Projects/assets/86758637/bfadd691-5e09-4fdd-9eaa-2e66af11765e)

### 3.1. Missing Values
Let's find out missing values in our dataset, To find the missing values we used isnull() function.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/148a47ba-57ed-4566-92ac-903d54cb672c)
![image](https://github.com/Lakindu1999/Projects/assets/86758637/ece5f20f-c7c4-4f0e-a790-7d94574c558d)

### 3.2. Dropping Column
With the above information missing values in the given data set. But there is an unnamed field in it. It should be removed.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/9b35809e-62f8-474d-9402-9cda7dd15b51)

### 3.3. Changing the data/variable types
Let’s change the variable types object (yes / no) to Boolean type true / false.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/f2871163-f111-4675-9803-6d19aca83611)

## 4. Exploratory Data Analysis
For my dataset analysis I am first going to check the correlation between the variables and plot a heat map for it.
Before exploring data, I have to drop some columns according to my task.

### 4.1. Correlation
Finding relation between variables using iloc function.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/486ac51f-8a1a-4f91-9137-05a6323f9677)

Creating hashmap to pinpoint the column in the dataframe for high correlation
![image](https://github.com/Lakindu1999/Projects/assets/86758637/798178dd-bd59-427a-81ee-54cf1f862cdb)
We can clearly observe that my target variable – G2.Port exhibits hightest correlation with ‘higher’ (want to take higher education) or not. And we also observe that least correlation is with ‘failures.Math’ and ‘failures.Port’ which might mean that either someone who failure math or port depend for G2.Port score.
We can observe that there are few other variables which shows correlation for example ‘Medu’, ‘Fedu’, ‘studytime’ with G2.Port and invers correlation between ‘Walc’, ‘Dalc’ which mean more alcoholic people had bad score for ‘G2.Port’.

### 4.2. Boxplot – Checking outliers
creating the boxplots for all variables to view and understand the outliers.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/fa1e2ed4-d0f4-459d-856a-549713150c94)

### 4.3. Barplots
For the first step we can visualize how student preform within ‘G2.Port’.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/c9fda868-2a03-4eb8-b07f-4521d37a8dd6)
Each bar corresponds to a specific grade and its height represents the number of students who received that grade. The most common grade appears to be 12, with over 100 students receiving it.

For target variable ‘G2.Port’ I would like to check that, are most want to take ‘higher’ education or not.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/3db2cae6-ce40-43ea-b884-e24954c5ed81)
Now we can see both school’s students like to go ahead with the Higher education for there future. According to this let’s compare the correlation between higher education among G2.Port to see how it various among them.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/4772212b-cd64-49f6-b3e3-45a5245b03b9)
This visualization provides a comparison of the Portuguese grades between students who want to pursue higher education and those who don’t. It appears that students who want to pursue higher education have a slightly higher average grade in Portuguese.

Let’s check which school has more average marks to G2.Port subject.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/efa92d97-cd8d-439b-868c-a4ffe3665779)
This visualization provides a comparison of the Portuguese grades between students from two different schools. It appears that students from Gabriel Pereira have a slightly higher average grade in Portuguese than those from Mousinho da Silveira. However, the difference between mean score of two schools not much value difference.

Let’s check average G2.Port Score over Gender.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/d5c735f7-5cbd-4226-a09f-7fc7171bf8d2)
This visualization provides a comparison of the Portuguese grades between female and male students. It appears that female students have a slightly higher average grade in Portuguese than male students. Male students who had score around 11 and female students scored nearly 12. According to the gender G2.Port score not strongly correlate.

Now we have to check the number of students in each school for our next few barplots.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/8e901703-b23c-490c-b3b2-135a823965b9)
The bar for GP is reaches up to approximately 600 on the y-axis, indicating a much larger student population at GP. The bar for MS is only reaches up to about 50 on the y-axis, indicating a significantly smaller student population at MS.

From above understanding we plot failure of Port and Math in both schools to understanding which school had more failure students.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/21ec7b6b-5517-4565-afd3-c8d625d529fd)
Both plots have bars representing the number of students who experienced 0, 1, 2 or more than 3 failures. In both plots, there are two colours for bars; blue represents school GP and orange represents school MS. For both subjects and schools, most students had zero failures as indicated by the tallest bars at position ‘0’. There are very few students with one failure in both subjects and almost none with two or more than three failures. This visualization provides a clear picture of the distribution of past class failures among students in the two schools for both subjects. Most students have not experienced any failures, which is a positive sign. We can see School GP had some more students who had failure 3 times for mathematics comparing to others.

We plot failure of Port and Math in both male and female to understanding which gender had more failure students.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/1d798196-dcd3-4151-929d-6ba7179f1dfc)
In both subjects, most students have zero past class failures; this category has the highest bars. For Portuguese class, over 300 females had zero failures while around less than half males had zero failures. Fewer than 50 males and females each had one failure. An extremely small number had two or more than three failures. For Math class, around less than half males had zero failure while over around less than half females also had zero failure. Fewer than around less than half males and females each had one failure. A small number had two or more than three failures. This visualization provides a clear picture of the distribution of past class failures among students in the two subjects for both genders. Most students have not experienced any failures, which is a positive sign.

From above understanding now we can plot a bar graph using G2.Port score among failure.math and failure.port.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/3721dcc7-c3a1-49fe-87dc-d4b542f02cc0)
The graphs are comparing “Failure Port among G2.Port” and “Failure Math among G2.Port”. This means they are showing how the number of past class failures in Port and Math subjects is distributed among different G2.Port scores. In both graphs, you can see that as ‘G2.Port’ scores increase, the number of students with past class failures decreases. This suggests that students with higher second period grades in port tend to have fewer past class failures. Most students have zero past class failures as indicated by the predominance of blue bars. This is a positive sign indicating good performance of most students. Students scoring between 10 to 15 in ‘G2.Port’ have notably fewer instances of past class failure. This further emphasizes the point that higher grades are associated with fewer failures.

We can observe that highest number of students count had very-low alcohol consumption, let’s visualise the same output below, in form of a plot for better understanding.
Let’s discuss regarding daily (Dalc) and weekly (Walc) alcohol consumption among a group of individuals.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/af7ab011-e860-44a4-83bb-405874631598)
In summary, the bar plots suggest that while daily alcohol consumption is generally low among the surveyed individuals, there is a more varied distribution of alcohol consumption on weekends. This could indicate that some individuals may consume more alcohol on weekends compared to weekdays.

Let’s see the relationship between students’ second period grades of port (G2.Port) and their workday (Dalc) and weekend (Walc) alcohol consumption levels.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/16956bab-70a4-4c8d-92b9-3148c98b59b1)
![image](https://github.com/Lakindu1999/Projects/assets/86758637/11c8bfa5-fa2a-4d0e-8145-576cac820446)
According to the scatter plots there have slight difference between daily alcohol consumption and weekly alcohol consumption among G2.Port score. I see who consumption more alcohol daily got bad score for G2.Port. But coming to the second scatter plot its not be significantly affected by their alcohol consumption levels.


### 4.4. Lineplots
For more analysis we can go for lineplots.
Let’s illustrates the relationship between students’ weekly study time and their second period grade in Port
![image](https://github.com/Lakindu1999/Projects/assets/86758637/c28e4e4b-79c2-4329-9b26-06d147b4498f)
The blue line on the graph depicts the relationship between study time and mean G2.Port score. It shows an increase in scores as study time increases up until category “3” (5 to 10 hrs), after which scores decrease. This suggests that studying for 5 to 10 hours per week may yield the best results in terms of the G2.Port score. However, studying for more than 10 hours per week seems to lead to a decrease in scores. This could be due to various factors such as burnout, stress, or lack of balance between study and relaxation.

After that we have 2 more variables (Fedu, Medu) left to visualize for more understanding.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/b9fedf1a-f63c-4dff-83a2-78e782824c5a)
![image](https://github.com/Lakindu1999/Projects/assets/86758637/c76001a3-3c7c-4a51-8815-0198d2b27c00)
The blue line on the first graph depicts the relationship between the father’s education level and the mean G2.Port score. It shows an increase in Mean G2.Port Score as Fedu continues to increase. This suggests that students whose fathers have higher levels of education tend to have higher grades in the second period of Port. The second graph depicts the relationship between the mother’s education level and the mean G2.Port score. It shows an initial plateau followed by a steady increase in Mean G2.Port Score as Medu increases beyond 2. This suggests that students whose mothers have higher levels of education tend to have higher grades in the second period of Port.


## 5. Modelling
Distributed the dependent and independent variables in x and y. My target variable for this dataset is 'G2.Port'. Once done I will split the dataset into training and test dataset. This is important because when we create machine learning models, the machine learns from the training data. It observes patterns, critical information’s and others, and then we use them to do prediction on the test dataset. The test dataset is used to test the accuracy of the model. The higher the accuracy will be, the better prediction of the model.

### 5.1. Drop Unnecessary columns before modeling
Now I finished my analysis part. For the first step of the modeling, we must drop unnecessary columns such as ‘address’, ‘reason’ and ‘guardian’ which that not needed for this modeling part. To prevent confusion, I have stored the dataset into a new dataframe after dropping.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/096be48f-d61a-4bbb-afff-5363ece6d096)

After that, I have encoded the categorical variable – ‘school’, ‘sex’ and ‘Fjob’ so that I converted to numarical values and does not cause error in model building.
Encoding categorical variable using ‘OneHotEncoding’ technique and change it to numerical value.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/71c5911e-0a5e-4d19-b419-cdda326c5f70)
![image](https://github.com/Lakindu1999/Projects/assets/86758637/8a7d972b-277a-4501-801f-c12b9e9981e5)

Encode categorical features as a one-hot numeric array. The input to this transformer should be an array-like of integers or strings, denoting the values taken on by categorical (discrete) features. The features are encoded using a one-hot (aka 'one-of-K' or 'dummy') encoding scheme.
This code will replace the ‘Fjob’, ‘school’ and ‘sex’ columns in your original DataFrame with new columns. Each new column corresponds to a unique category in the ‘Fjob’, ‘school’ and ‘sex’ column. The values in these new columns are 0s and 1s, indicating the absence or presence of the category in the original data.
For Example: Fjob - father’s job we have 5 categories “teacher”, “health” care related, “services”, “at_home” or “other”. Now we can create column with 0 or 1 which means 0 is not and 1 is yes like that.
•
If ‘Fjob’ is teacher then it looks like, Fjob_teacher Fjob_services Fjob_health Fjob_home Fjob_other 1 0 0 0 0
If ‘Fjob’ is home then it looks like, Fjob_teacher Fjob_services Fjob_health Fjob_home Fjob_other 0 0 0 1 0


## 5.2. Regression
We now do prediction for our target variable- G2.Port by applying into various regression models.
Before that we creating a new dataframe to store including R2_score, MAE and MSE for compare different Regression models at the end.
For all the Regression model we calculate R2 square, the MAE and MSE.
•
MAE (Mean Absolute Error) - A statistics that reveals the mean absolute difference between a dataset's real values and its predicted values. The better a model matches a dataset, the lower the MAE will be
•
MSE (Mean Square Error) - The term Mean Squared Error (MSE) refers to the square of the differences between the actual and estimated values in statistics.
•
R-Squared - In a regression model, R-Squared (also known as R2 or the coefficient of determination) is a statistical measure which is used to determine the proportion of variance in the dependent and independent variables, as such that this variance in depend variable can be explained in the independent ones.


### 5.2.1. Multiple Linear Regression
Let first try applying Multiple Linear Regression model.
We forecast the value of a variable in a linear regression analysis using the value of another variable. In this study, one or more independent variables that are the best predictors of the value of the dependent variable are used to estimate the coefficients of the linear equation. Our target variable, G2.Port, is the dependent variable in this situation.
The differences between expected and actual output values are minimized by linear regression by fitting a straight line or surface.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/4b04741d-e0f6-47b1-91cd-4cdc491b4a04)
With above evaluation data, it can be seen that Multiple Linear Regression model has a R2 score of 0.271(27%) with testing data and it is not a good score.
Coming to the MAE and MSE respectively 1.58 and 4.18 are both not good enough.


### 5.2.2. Decision Tree Regression
Now we implement our dataset in the Decision Tree Regression model.
A regression tree, is used to predict continuous valued outputs rather than discrete outputs. In order to do the prediction and provide findings that are more precisely tailored to the non-linear distribution of the dataset, Decision Tree Regression aids in the division of a dataset into smaller subgroups.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/21b2ac64-0aa4-450e-9e02-2ada7ef46922)
With above evaluation data, it can be seen that Decision Tree Regression model has a R2 score of 0.741(74%) with testing data and it is a good score comparing to the previous model.
Coming to the MAE and MSE respectively 0.52 and 1.47 both are good enough values.


### 5.2.3. Random Forest Regression
Let’s implement our dataset in the Random Forest Regression model.
Random Forest - With the help of several decision trees and a method known as Bootstrap and Aggregation, also referred to as bagging, this model is an ensemble methodology which is capable of handling both regression and classification tasks. This method's fundamental principle is to integrate several decision trees and to get the final result rather than depending solely on one decision tree.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/3fbe3984-7ffd-4502-a727-60a835eb9b05)
With above evaluation data, it can be seen that Random Forest Regression model has a R2 score of 0.793(79%) with testing data and it is a good score among regression models.
Coming to the MAE and MSE respectively 0.70 and 1.18 both are good enough values.


### 5.2.4. Support Vector Regression
Now let’s implement our dataset in the Support Vector Regression model.
SVR - Both linear and non-linear regressions are supported by this approach. The Support Vector Machine is the basis for how this approach operates. In contrast to SVM, which is used to predict discrete categorical labels, SVR is a regressor which is used to forecast continuous ordered variables. In this manner SVR and SVM differ from one another.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/bd04e877-eda9-4114-aa4d-a1624f4df14b)
With above evaluation data, it can be seen that Support Vector Regression model has a R2 score of 0.216(21%) with testing data and it is not a good score.
Coming to the MAE and MSE respectively 1.64 and 4.50 are both not good enough for our model.


### 5.2.5. Comparison between different Regression Algorithm
At the beginning of the modelling, we created a dataframe to store the accuracy now we can easily compare every regression algorithm models we used and find a best suitable regression model for our task.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/0d7c23a9-d673-4857-bb17-8d0a420873b9)
Based on the table, the Random Forest Regression model would be the most suitable choice for our task. It has the highest R2 score of 0.793534(79%) among the models listed, indicating a better fit to the data.
Although the Decision Tree Regression model has slightly lower error metrics (MAE = 0.520468; MSE = 1.479532) compared to Random Forest Regression (MAE = 0.703528; MSE = 1.186575), the higher R2 score of the Random Forest model suggests it is more effective at explaining the variance in our data. Therefore, considering these factors, the Random Forest Regression model appears to be the best choice for our task.



## 5.3. Classification
We now do prediction for our target variable- G2.Port by applying into various classification models.
Before start the classification modelling we have to bin the target variable G2.Port into 4 categories in such a way that the resulting bins contain roughly equal number of cases. Apart from that we distributing dependent and independent variable, Splitting the dataset into Training and Test data and creating a new dataframe to store accuracy score for compare different classification models at the end.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/f0cd3d2c-bea5-4307-8aec-6692bf4ae435)
In this code, KBinsDiscretizer is used to bin the G2.Port column into 4 categories. The strategy='quantile' parameter ensures that the bins are defined to be approximately equal frequency. The rest of the code remains the same. You can see the count value of each 4 categories roughly equal numbers.
For all the Classification model we calculate Accuracy, the Macro Avg and Weighted Avg.
•
Precision - The ratio of true positive predictions to the total number of positive predictions.
•
Recall - the ratio of true positive predictions to the total number of actual positive instances.
•
F1-score - The harmonic means of precision and recall. It provides a balance between precision and recall.
•
Support - Represents the number of instances in each class in the test dataset.
•
Accuracy - The overall accuracy of the model, which is the ratio of correctly predicted instances to the total number of instances.
•
Macro Avg - Provides the average of precision, recall, and F1-score across all classes.
•
Weighted Avg - Provides the weighted average of precision, recall, and F1-score, where the weights are based on the number of instances in each class.


### 5.3.1. Decision Tree Classifier
The Decision Tree algorithm selects the best attribute to split the data based on a metric such as entropy or Gini impurity, which measures the level of impurity or randomness in the subsets. The goal is to find the attribute that maximizes the information gain or the reduction in impurity after the split.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/e7aebfe3-5f1a-44a7-983c-6f88c8f36761)
With above evaluation data, it can be seen that Decision Tree Classifier model has an accuracy of 0.78(78%) with testing data and it is a good score.
Calculate confusion matrix for decision tree classifier.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/7de40071-273b-4fd8-a319-8f8435618315)


### 5.3.2. Multiple Logistic Regression
Multiple Logistic regression attempts to classify observations from a dataset into distinct categories. It’s mainly used for binary classification; it uses a linear combination of the observed features and some problem-specific parameters to estimate the probability of each particular value of the dependent variable.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/888a54b9-c3ec-47b7-a33f-291d3524e2cb)
With above evaluation data, it can be seen that Multiple Logistic Regression model has an accuracy of 0.53(53%) with testing data and it is not a good score.
Calculate confusion matrix for Logistic Regression.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/99a509b8-c96f-4e9f-a396-ca01b80a2714)


### 5.3.3. KNN
In classification, KNN assigns a class label based on a majority vote of the nearest neighbors. In regression, it predicts a value based on the average of the values of its nearest neighbors. For that problem I put 5 neighbors for the predict the model.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/679d0e4c-b69e-4974-b6af-51fd8f522613)
With above evaluation data, it can be seen that KNN model has an accuracy of 0.52(52%) with testing data and it is not a good score.
Calculate confusion matrix for KNN.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/07acf917-1cdd-4cd2-afb4-211d8a7c59aa)


### 5.3.4. Gradient Boosting Classifier
Gradient Boosting Classifiers (GBC) are a type of machine learning algorithm used for both classification and regression problems; The idea of boosting came out of the idea of whether a weak learner can be modified to become better. GBCs work by building simpler (weak) prediction models sequentially where each model tries to predict the error left over by the previous model. Despite their potential for overfitting, they are widely used in various fields due to their predictive power.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/46e6fadf-c8e5-4aef-a835-25efd32bff6f)
With above evaluation data, it can be seen that Gradient Boosting Classifier model has an accuracy of 0.80(80%) with testing data and it is a good score.
Calculate confusion matrix for Gradient Boosting Classifier.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/85c306ce-651e-4467-bc87-72723f362c24)


### 5.3.5. Random Forest
Random Forest - With the help of several decision trees and a method known as Bootstrap and Aggregation, also referred to as bagging, this model is an ensemble methodology which is capable of handling both regression and classification tasks. This method's fundamental principle is to integrate several decision trees and to get the final result rather than depending solely on one decision tree.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/fcf12c21-0784-4442-8c41-4ee4d00d1bf6)
With above evaluation data, it can be seen that Random Forest model has an accuracy of 0.85(85%) with testing data and it is a good score.
Calculate confusion matrix for Random Forest.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/fadc39da-8753-459d-a81a-cb2cfa6d5181)


### 5.3.6. Comparison between different Classification Algorithm
At the beginning of the modelling, we created a dataframe to store the accuracy now we can easily compare every classification models we used and find a best suitable classification model for our task.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/4b1820ad-ece3-4645-bf57-12b44adcf1fa)
If you prioritize precision, especially for classes 1.0 and 3.0, and slightly higher overall accuracy, the Random Forest Classifier might be a suitable choice.
If you prioritize recall, especially for class 2.0, and you find the overall F1-scores more important, the Gradient Boosting Classifier might be preferred.
Both models seem to perform reasonably well, and the choice between them can be influenced by factors such as interpretability, computational efficiency, and the specific goals of your task. If interpretability is important, Random Forests are often easier to understand. If you are looking for a model with high predictive accuracy and are willing to invest more in computation, Gradient Boosting can be a good choice.


### 5.3.7. Cross-Validation for Preformed Model
Grid Search Cross Validation helps prevent overfitting by assessing model performance on different data subsets and optimizes hyperparameters, ensuring the model generalizes well to new, unseen data. It is a widely used technique to improve the robustness and effectiveness of machine learning models.
![image](https://github.com/Lakindu1999/Projects/assets/86758637/d4679fea-ea1d-4472-b488-5b7f9b80a8ef)
With these tuned hyperparameters, the Random Forest model achieved an accuracy of approximately 84.21% during cross-validation. This indicates that the model, after hyperparameter tuning, is performing well on the validation sets and is likely to generalize effectively to new, unseen data.


## 6. Results and Conclusions
In conclusion, The Random Forest Regression model demonstrates superior performance compared to Multiple Linear Regression, Decision Tree Regression, and Support Vector Regression, as evidenced by its higher R2 score (0.7935) and relatively lower Mean Absolute Error (0.7035) and Mean Squared Error (1.1866). This suggests that the Random Forest Regression, leveraging an ensemble approach, better captures the complex relationships within the data, providing a more accurate prediction of the target variable. Therefore, the Random Forest Regression model is recommended for the given regression task. In conclusion, The Random Forest Classifier outperforms other models in terms of accuracy. It seems to be a promising choice for the given classification task, providing the highest accuracy among the models evaluated. Gradient Boost Classifier also performs well, but Random Forest stands out as the top performer in this comparison.


## References
(scikit-learn: machine learning in Python — scikit-learn 1.4.0 documentation, no date)
(Machine Learning Tutorial, no date)
(iloc() Function in Python, no date)
(Confusion Matrix Visualization. How to add a label and percentage to a… | by Dennis T | Medium, no date)
(Python Machine Learning - Cross Validation, no date)
(iloc() Function in Python - Scaler Topics, no date)
