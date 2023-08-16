#!/usr/bin/env python
# coding: utf-8

# # **Capstone project: Providing data-driven suggestions for HR**

# ## **Pace: Plan**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.
# 
# In this stage, consider the following:

# ### Understand the business scenario and problem
# 
# The HR department at Salifort Motors wants to take some initiatives to improve employee satisfaction levels at the company. They collected data from employees, but now they don’t know what to do with it. They refer to you as a data analytics professional and ask you to provide data-driven suggestions based on your understanding of the data. They have the following question: what’s likely to make the employee leave the company?
# 
# Your goals in this project are to analyze the data collected by the HR department and to build a model that predicts whether or not an employee will leave the company.
# 
# If you can predict employees likely to quit, it might be possible to identify factors that contribute to their leaving. Because it is time-consuming and expensive to find, interview, and hire new employees, increasing employee retention will be beneficial to the company.

# ## Step 1. Imports
# 
# *   Import packages
# *   Load dataset
# 
# 

# In[1]:


# Import packages

# For data manipulation
import numpy as np
import pandas as pd

# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)

# For data modeling
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# For metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

# For saving models
import pickle


# ### Load dataset

# In[2]:


# Load dataset into a dataframe
df0 = pd.read_csv("HR_capstone_dataset.csv")

# Display first few rows of the dataframe 
df0.head()


# ## Step 2. Data Exploration (Initial EDA and data cleaning)
# 
# - Understand your variables
# - Clean your dataset (missing data, redundant data, outliers) 
# 
# 

# In[3]:


# Gather basic information about the data
df0.info()


# ### Gather descriptive statistics about the data

# In[4]:


# Gather descriptive statistics about the data
df0.describe()


# ### Rename columns

# As a data cleaning step, rename the columns as needed. Standardize the column names so that they are all in `snake_case`, correct any column names that are misspelled, and make column names more concise as needed.

# In[5]:


# Display all column names
df0.columns


# In[6]:


# Rename columns as needed
### YOUR CODE HERE ### 
df0 = df0.rename(columns={'Work_accident': 'work_accident',
                          'average_montly_hours': 'average_monthly_hours',
                          'time_spend_company': 'tenure',
                          'Department': 'department'})

# Display all column names after the update
### YOUR CODE HERE ### 
df0.columns


# ### Check missing values

# Check for any missing values in the data.

# In[7]:


# Check for missing values
df0.isna().sum()


# There are no missing values in the data.

# ### Check duplicates

# Check for any duplicate entries in the data.

# In[9]:


# Check for duplicates
### YOUR CODE HERE ###
df0.duplicated().sum()


# 3,008 rows contain duplicates.

# In[10]:


# Inspect some rows containing duplicates as needed
### YOUR CODE HERE ###
df0[df0.duplicated()].head()


# 
# The provided output displays the initial occurrences of rows that reappear later in the dataframe as duplicates. Assessing the credibility of these entries involves evaluating the possibility of two employees independently providing identical responses across all columns. 
# While a complex likelihood analysis using Bayes' theorem could be conducted, it appears unnecessary in this scenario. Given the presence of numerous continuous variables spanning 10 columns, the probability of these observations being genuine seems quite low. Consequently, a prudent course of action would involve discarding these duplicated rows.

# In[11]:


# Drop duplicates and save resulting dataframe in a new variable as needed
df1 = df0.drop_duplicates(keep='first')

# Display first few rows of new dataframe as needed
df1.head()


# ### Check outliers

# Check for outliers in the data.

# In[12]:


# Create a boxplot to visualize distribution of `tenure` and detect any outliers
plt.figure(figsize=(6,6))
plt.title('Boxplot to detect outliers for tenure', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df1['tenure'])
plt.show()


# The boxplot above shows that there are outliers in the `tenure` variable. 
# 
# It would be helpful to investigate how many rows in the data contain outliers in the `tenure` column.

# # pAce: Analyze Stage
# - Perform EDA (analyze relationships between variables) 
# 
# 

# In[13]:


# Determine the number of rows containing outliers

# Compute the 25th percentile value in `tenure`
percentile25 = df1['tenure'].quantile(0.25)

# Compute the 75th percentile value in `tenure`
percentile75 = df1['tenure'].quantile(0.75)

# Compute the interquartile range in `tenure`
iqr = percentile75 - percentile25

# Define the upper limit and lower limit for non-outlier values in `tenure`
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

# Identify subset of data containing outliers in `tenure`
outliers = df1[(df1['tenure'] > upper_limit) | (df1['tenure'] < lower_limit)]

# Count how many rows in the data contain outliers in `tenure`
print("Number of rows in the data containing outliers in `tenure`:", len(outliers))


# ## Step 2. Data Exploration (Continue EDA)
# 
# Begin by understanding how many employees left and what percentage of all employees this figure represents.

# In[14]:


# Get numbers of people who left vs. stayed
print(df1['left'].value_counts())
print()

# Get percentages of people who left vs. stayed
print(df1['left'].value_counts(normalize=True))


# ### Data visualizations

# Now, examine variables that you're interested in, and create plots to visualize relationships between variables in the data.

# To begin, a stacked boxplot can effectively illustrate the distribution of average_monthly_hours across different number_project categories, facilitating a comparison between employees who remained with the company and those who departed.
# 
# While box plots provide valuable insights into data distribution, they may lack clarity regarding sample sizes. To address this, a stacked histogram depicting the distribution of number_project for both employee groups (retained and departed) can further enhance our understanding of the data.

# In[16]:


# Create a plot as needed 

# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Create boxplot showing `average_monthly_hours` distributions for `number_project`, comparing employees who stayed versus those who left
sns.boxplot(data=df1, x='average_monthly_hours', y='number_project', hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Monthly hours by number of projects', fontsize='14')

# Create histogram showing distribution of `number_project`, comparing employees who stayed versus those who left
tenure_stay = df1[df1['left']==0]['number_project']
tenure_left = df1[df1['left']==1]['number_project']
sns.histplot(data=df1, x='number_project', hue='left', multiple='dodge', shrink=2, ax=ax[1])
ax[1].set_title('Number of projects histogram', fontsize='14')

# Display the plots
plt.show()


# It's plausible that individuals handling more projects might also dedicate longer hours, a trend evident in this analysis. The mean work hours for both the 'stayed' and 'left' groups exhibit an upward trajectory with an increasing number of projects undertaken. Yet, this visualization reveals several intriguing observations:
# 
# * The 'left' category encompasses two distinct employee subsets: (A) those who logged notably fewer hours than their peers with matching project counts, and (B) those who contributed substantially more. For group A, the possibility of termination or imminent departure looms. It's conceivable that this subset contains individuals who, anticipating their exit, were assigned reduced hours. Conversely, group B likely comprises voluntary departures, possibly indicating a significant role in their project's success.
# 
# * Noteworthy is the unanimous departure of all individuals engaged in seven projects. Additionally, the interquartile ranges of the seven and six-project 'left' categories—hovering around 255–295 hours weekly—surpass other groups by a significant margin.
# 
# * An optimal project count materializes within the 3–4 range, where the 'left' to 'stayed' ratio diminishes significantly, suggesting a favorable equilibrium.
# 
# * Assuming a 40-hour workweek and two weeks' annual vacation, the average monthly work hours for Monday–Friday employees equate to approximately 166.67 hours. Interestingly, excluding the two-project workers, every group—'stayed' or not—accumulates considerably more hours. This prompts the notion of potential overexertion among employees.
# 
# The forthcoming step involves confirming the categorical exodus of employees engaged in seven projects.

# In[17]:


# Get value counts of stayed/left for employees with 7 projects
df1[df1['number_project']==7]['left'].value_counts()


# This confirms that all employees with 7 projects did leave. 
# 
# Next, you could examine the average monthly hours versus the satisfaction levels. 

# In[18]:


# Create a plot as needed 

# Create scatterplot of `average_monthly_hours` versus `satisfaction_level`, comparing employees who stayed versus those who left
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df1, x='average_monthly_hours', y='satisfaction_level', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by last evaluation score', fontsize='14');


# The presented scatterplot highlights a notable cluster of employees devoting approximately 240 to 315 monthly hours, equivalent to over 75 hours weekly throughout a year. Remarkably, this workload surpasses conventional bounds. A correlation emerges with their satisfaction levels, hovering near zero—a connection potentially underpinning their departure.
# 
# Furthermore, the plot discerns another departing cohort, characterized by comparatively standard working hours. Nonetheless, their satisfaction merely registers around 0.4. Delving into the reasons becomes intricate; plausible speculation revolves around perceived pressure to conform to extended work hours, given the prevalent trend, inevitably dampening their contentment.
# 
# Lastly, a distinct assemblage emerges, clocking in at approximately 210 to 280 monthly hours, paired with satisfaction levels spanning 0.7 to 0.9. The peculiar shape of these distributions raises eyebrows, hinting at possible data manipulation or the introduction of synthetic data, warranting careful consideration.
# 
# It is crucial to note the unconventional distribution patterns, serving as potential indicators of data irregularities or fabricated information.

# For the next visualization, it might be interesting to visualize satisfaction levels by tenure.

# In[19]:


# Create a plot as needed 

# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Create boxplot showing distributions of `satisfaction_level` by tenure, comparing employees who stayed versus those who left
sns.boxplot(data=df1, x='satisfaction_level', y='tenure', hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Satisfaction by tenure', fontsize='14')

# Create histogram showing distribution of `tenure`, comparing employees who stayed versus those who left
tenure_stay = df1[df1['left']==0]['tenure']
tenure_left = df1[df1['left']==1]['tenure']
sns.histplot(data=df1, x='tenure', hue='left', multiple='dodge', shrink=5, ax=ax[1])
ax[1].set_title('Tenure histogram', fontsize='14')

plt.show();


# Several key insights can be gleaned from the provided plot:
# 
# 1. Departing employees can be broadly categorized into two groups: dissatisfied individuals with shorter employment spans and highly content individuals with moderate tenures.
#    
# 2. Notably, those who departed after four years exhibit an unusually low satisfaction level. This anomaly prompts an investigation into potential shifts in company policies that might have influenced employees around the four-year mark.
# 
# 3. The employees with the lengthiest tenures appear to have remained with the company. Remarkably, their satisfaction levels align closely with those of newer employees who chose to stay, suggesting a positive correlation between satisfaction and retention.
# 
# 4. The histogram highlights a scarcity of individuals with extended tenures. This scarcity could indicate that these longer-tenured employees might hold higher-ranking positions with commensurate compensation.
# 
# As the subsequent analytical stride, computing the mean and median satisfaction scores for departing and retained employees would provide valuable quantitative insights into the overall satisfaction trends and potential differences between the two groups.

# In[20]:


# Calculate mean and median satisfaction scores of employees who left and those who stayed
df1.groupby(['left'])['satisfaction_level'].agg([np.mean,np.median])


# As anticipated, the calculated mean and median satisfaction scores for departing employees indeed exhibit lower values compared to those who chose to stay. Notably, within the group of retained employees, it's intriguing to observe that the mean satisfaction score slightly trails behind the median score. This observation suggests a potential leftward skew in the distribution of satisfaction levels among the retained workforce.
# 
# Moving forward, an examination of salary levels across varying tenures would provide valuable insights into the compensation structure and potential correlations with employee retention and satisfaction.

# In[21]:


# Create a plot as needed 

# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Define short-tenured employees
tenure_short = df1[df1['tenure'] < 7]

# Define long-tenured employees
tenure_long = df1[df1['tenure'] > 6]

# Plot short-tenured histogram
sns.histplot(data=tenure_short, x='tenure', hue='salary', discrete=1, 
             hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.5, ax=ax[0])
ax[0].set_title('Salary histogram by tenure: short-tenured people', fontsize='14')

# Plot long-tenured histogram
sns.histplot(data=tenure_long, x='tenure', hue='salary', discrete=1, 
             hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.4, ax=ax[1])
ax[1].set_title('Salary histogram by tenure: long-tenured people', fontsize='14');


# The visualizations presented earlier indicate that employees with longer tenures weren't predominantly composed of higher-paid individuals.
# 
# To further the analysis, an exploration into the potential correlation between extended working hours and elevated evaluation scores could provide valuable insights. A scatterplot portraying the relationship between `average_monthly_hours` and `last_evaluation` would be an appropriate next step.

# In[22]:


# Create a plot as needed 
# Create scatterplot of `average_monthly_hours` versus `last_evaluation`
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df1, x='average_monthly_hours', y='last_evaluation', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by last evaluation score', fontsize='14');


# The scatterplot above unveils several significant insights:
# 
# 1. Distinct groups of departing employees emerge: Firstly, individuals who were burdened with excessive work hours and yet exhibited exceptional performance; and secondly, those who operated slightly below the typical monthly average of 166.67 hours, accompanied by lower evaluation scores.
# 
# 2. An evident correlation emerges between the amount of hours worked and the corresponding evaluation score.
# 
# 3. While not heavily represented, the upper-left quadrant of the scatterplot is notably occupied by a relatively small proportion of employees. This implies that extended work hours alone do not guarantee a favorable evaluation score.
# 
# 4. A prevailing trend is observed, where a majority of employees within the company consistently surpass the 167-hour monthly threshold.
# 
# Shifting focus, the subsequent analysis seeks to ascertain whether employees who logged exceptionally long hours were subject to promotions within the past five years.

# In[23]:


# Create a plot as needed 

# Create plot to examine relationship between `average_monthly_hours` and `promotion_last_5years`
plt.figure(figsize=(16, 3))
sns.scatterplot(data=df1, x='average_monthly_hours', y='promotion_last_5years', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by promotion last 5 years', fontsize='14');


# The presented plot reveals the following insightful patterns:
# 
# 1. A notably sparse number of employees who were promoted in the last five years chose to leave the company.
# 2. Similarly, a scarcity of employees who logged the highest number of hours also received promotions.
# 3. Strikingly, every departing employee was found among those who worked the longest hours.
# 
# With this context in mind, the subsequent analysis aims to shed light on the distribution of departing employees across different departments.

# In[24]:


# Display counts for each department
df1["department"].value_counts()


# In[27]:


# Create a plot as needed
# Create stacked histogram to compare department distribution of employees who left to that of employees who didn't
plt.figure(figsize=(11, 8))
sns.histplot(data=df1, x='department', hue='left', discrete=1,
             hue_order=[0, 1], multiple='dodge', shrink=.5)
plt.xticks(rotation=45) 
plt.title('Counts of stayed/left by department', fontsize=14)
plt.tight_layout()  
plt.show()


# No particular department exhibits a substantial variance in the proportion of employees who departed in comparison to those who remained, indicating a relatively uniform distribution across departments.
# 
# To conclude this analysis, exploring potential robust correlations between various variables within the dataset would be a prudent step forward.

# In[30]:


# Create a plot as needed 
# Select only numeric columns for correlation calculation
numeric_columns = df0.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix
correlation_matrix = numeric_columns.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(correlation_matrix, vmin=-1, vmax=1, annot=True, cmap=sns.color_palette("vlag", as_cmap=True))
plt.title('Correlation Heatmap', fontsize=14)
plt.show()


# The correlation heatmap validates several significant relationships within the data:
# 
# 1. There exists a positive correlation among the number of projects, average monthly hours, and evaluation scores, indicating that employees who engage in more projects and invest greater hours tend to receive higher evaluation scores.
# 
# 2. A noteworthy negative correlation emerges between an employee's decision to leave and their satisfaction level. This suggests that higher dissatisfaction levels are associated with a greater likelihood of departure.
# 
# These observations underscore the complex interplay between various factors influencing employee behavior, satisfaction, and retention.

# ### Insights

# It appears that employees are leaving the company as a result of poor management. Leaving is tied to longer working hours, many projects, and generally lower satisfaction levels. It can be ungratifying to work long hours and not receive promotions or good evaluation scores. There's a sizeable group of employees at this company who are probably burned out. It also appears that if an employee has spent more than six years at the company, they tend not to leave. 

# # paCe: Construct Stage
# - Determine which models are most appropriate
# - Construct the model 
# - Confirm model assumptions
# - Evaluate model results to determine how well your model fits the data
# 

# ## Reviewing Model Assumptions for Logistic Regression
# 
# **Assumption 1: Categorical Outcome Variable**
# The logistic regression model assumes that the outcome variable is categorical in nature, typically binary (e.g., yes/no, 1/0). This is consistent with the nature of logistic regression, which is designed for binary classification problems.
# 
# **Assumption 2: Independence of Observations**
# The observations in the dataset should be independent of each other. This assumption ensures that the observations are not influenced by each other and that there is no serial correlation.
# 
# **Assumption 3: No Severe Multicollinearity Among X Variables**
# The model assumes that there is no severe multicollinearity among the predictor variables (X variables). High multicollinearity can lead to unstable coefficient estimates and reduced interpretability.
# 
# **Assumption 4: No Extreme Outliers**
# Extreme outliers in the data can disproportionately influence the model's coefficients and predictions. It's important to check for and address outliers, as they can impact the model's performance.
# 
# **Assumption 5: Linear Relationship Between X Variables and Logit of Outcome Variable**
# Logistic regression assumes a linear relationship between the predictor variables and the logit of the outcome variable. This assumption should be assessed to ensure that the relationship holds.
# 
# **Assumption 6: Sufficiently Large Sample Size**
# A sufficiently large sample size is needed to ensure the stability and reliability of the parameter estimates. The sample size should be large enough to meet the assumptions of asymptotic normality.
# 
# Ensuring that these assumptions are met is crucial for the validity and reliability of the logistic regression model and its results. It's important to assess and address violations of these assumptions to build a robust and accurate model.

# ## Step 3. Model Building, Step 4. Results and Evaluation
# - Fit a model that predicts the outcome variable using two or more independent variables
# - Check model assumptions
# - Evaluate the model

# ### Identify the type of prediction task.

# Certainly, let me rephrase that for clarity:
# 
# Your objective is to predict whether an employee will remain with or leave the company, which involves a classification task. Specifically, this falls under the category of binary classification since the outcome variable, denoted as `left`, assumes two possible values: 1 (indicating that the employee left) or 0 (indicating that the employee stayed).

# ### Identify the types of models most appropriate for this task.

# Given the categorical nature of the variable you intend to predict (employee departure from the company), you have two viable options for building a predictive model:
# 
# Logistic Regression Model: Logistic regression is a suitable choice for binary classification tasks. It's a well-established method that can provide insights into the relationships between predictor variables and the likelihood of an employee leaving the company.
# 
# Tree-Based Machine Learning Model: Tree-based models, such as decision trees or random forests, are powerful tools for classification tasks. They can capture complex interactions between variables and provide predictive accuracy.

# ### Modeling Approach A: Logistic Regression Model
# 
# This approach covers implementation of Logistic Regression.

# #### Logistic regression
# Note that binomial logistic regression suits the task because it involves binary classification.

# Prior to dividing the dataset, it's essential to encode the non-numeric variables. This involves handling two variables: `department` and `salary`.
# 
# Regarding the `department` variable, which falls under the categorical type, an effective approach for modeling involves creating dummy variables.
# 
# On the other hand, `salary` also lies within the categorical realm, but it is of the ordinal nature. Given the presence of a discernible hierarchy among its categories, opting against the creation of dummy variables is advisable. Instead, it's more appropriate to represent the distinct levels numerically, using the range from 0 to 2.

# In[32]:


# Copy the dataframe
df_enc = df1.copy()

# Encode the `salary` column as an ordinal numeric category
df_enc['salary'] = (
    df_enc['salary'].astype('category')
    .cat.set_categories(['low', 'medium', 'high'])
    .cat.codes
)

# Dummy encode the `department` column
df_enc = pd.get_dummies(df_enc, drop_first=False)

# Display the new dataframe
df_enc.head()


# Create a heatmap to visualize how correlated variables are. Consider which variables you're interested in examining correlations between.

# In[34]:


# Create a heatmap to visualize how correlated variables are
plt.figure(figsize=(8, 6))
sns.heatmap(df_enc[['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours', 'tenure']]
            .corr(), annot=True, cmap="crest")
plt.title('Heatmap of the dataset')
plt.show()


# Create a stacked bart plot to visualize number of employees across department, comparing those who left with those who didn't.

# In[35]:


# Create a stacked bart plot to visualize number of employees across department, comparing those who left with those who didn't
# In the legend, 0 (purple color) represents employees who did not leave, 1 (red color) represents employees who left
pd.crosstab(df1['department'], df1['left']).plot(kind ='bar',color='mr')
plt.title('Counts of employees who left versus stayed across department')
plt.ylabel('Employee count')
plt.xlabel('Department')
plt.show()


# Given the sensitivity of logistic regression to outliers, it's a prudent step to proceed by removing the outliers detected earlier in the `tenure` column. Doing so will help mitigate potential distortions and enhance the reliability of the model's results and predictions.

# In[36]:


# Select rows without outliers in `tenure` and save resulting dataframe in a new variable
df_logreg = df_enc[(df_enc['tenure'] >= lower_limit) & (df_enc['tenure'] <= upper_limit)]

# Display first few rows of new dataframe
df_logreg.head()


# Separate the outcome variable, which serves as the variable you intend your model to predict.

# In[37]:


# Isolate the outcome variable
y = df_logreg['left']

# Display first few rows of the outcome variable
y.head() 


# Choose the features that you intend to incorporate into your model. Consider which variables are likely to contribute to predicting the outcome variable, `left`.

# In[38]:


# Select the features you want to use in your model
X = df_logreg.drop('left', axis=1)

# Display the first few rows of the selected features 
X.head()


# Partition the data into a training set and a testing set. Ensure that you employ stratification based on the values within `y` to account for the imbalanced nature of the classes.

# In[39]:


# Split the data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


# Construct a logistic regression model and fit it to the training dataset.

# In[40]:


# Construct a logistic regression model and fit it to the training dataset
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)


# Test the logistic regression model: use the model to make predictions on the test set.

# In[42]:


# Use the logistic regression model to get predictions on the test set
y_pred = log_clf.predict(X_test)


# Create a confusion matrix to visualize the results of the logistic regression model. 

# In[44]:


# Compute values for confusion matrix
log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, 
                                  display_labels=log_clf.classes_)

# Plot confusion matrix
log_disp.plot(values_format='')

# Display plot
plt.show()


# The upper-left quadrant displays the number of true negatives.
# The upper-right quadrant displays the number of false positives.
# The bottom-left quadrant displays the number of false negatives.
# The bottom-right quadrant displays the number of true positives.
# 
# True negatives: The number of people who did not leave that the model accurately predicted did not leave.
# 
# False positives: The number of people who did not leave the model inaccurately predicted as leaving.
# 
# False negatives: The number of people who left that the model inaccurately predicted did not leave
# 
# True positives: The number of people who left the model accurately predicted as leaving
# 
# A perfect model would yield all true negatives and true positives, and no false negatives or false positives.

# Create a classification report that includes precision, recall, f1-score, and accuracy metrics to evaluate the performance of the logistic regression model.

# Check the class balance in the data. In other words, check the value counts in the `left` column. Since this is a binary classification task, the class balance informs the way you interpret accuracy metrics.

# In[46]:


df_logreg['left'].value_counts(normalize=True)


# The data distribution is approximately 83% for one class and 17% for the other, indicating a moderate imbalance. While not severely skewed, if the imbalance were more pronounced, resampling techniques could be employed to enhance class balance. However, in this scenario, the existing class distribution can be retained for model evaluation without necessitating adjustments.

# In[47]:


# Create classification report for logistic regression model
target_names = ['Predicted would not leave', 'Predicted would leave']
print(classification_report(y_test, y_pred, target_names=target_names))


# The classification report above shows that the logistic regression model achieved a precision of 79%, recall of 82%, f1-score of 80% (all weighted averages), and accuracy of 82%. However, if it's most important to predict employees who leave, then the scores are significantly lower.

# ### Modeling Approach B: Tree-based Model
# This approach covers implementation of Decision Tree and Random Forest. 

# Isolate the outcome variable.

# In[48]:


# Isolate the outcome variable
y = df_enc['left']

# Display the first few rows of `y`
y.head()


# Select the features. 

# In[49]:


# Select the features
X = df_enc.drop('left', axis=1)

# Display the first few rows of `X`
X.head()


# Split the data into training, validating, and testing sets.

# In[50]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


# #### Decision tree 

# Construct a decision tree model and set up cross-validated grid-search to exhuastively search for the best model parameters.

# In[51]:


# Instantiate model
tree = DecisionTreeClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, 8, None],
             'min_samples_leaf': [2, 5, 1],
             'min_samples_split': [2, 4, 6]
             }

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
tree1 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')


# Fit the decision tree model to the training data.

# In[52]:


get_ipython().run_cell_magic('time', '', 'tree1.fit(X_train, y_train)\n')


# Identify the optimal values for the decision tree parameters.

# In[53]:


# Check best parameters
tree1.best_params_


# Identify the best AUC score achieved by the decision tree model on the training set.

# In[54]:


# Check best AUC score on CV
tree1.best_score_


# The robust AUC score indicates the model's exceptional ability to predict employee attrition accurately.
# 
# Moving forward, you can formulate a function that facilitates the extraction of scores from the grid search results. This function will streamline the process of accessing and analyzing various performance metrics.

# In[55]:


def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
        model_name (string): what you want the model to be called in the output table
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, accuracy, or auc
  
    Returns a pandas df with the F1, recall, precision, accuracy, and auc scores
    for the model with the best mean 'metric' score across all validation folds.  
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'auc': 'mean_test_roc_auc',
                   'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy'
                  }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract Accuracy, precision, recall, and f1 score from that row
    auc = best_estimator_results.mean_test_roc_auc
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
  
    # Create table of results
    table = pd.DataFrame()
    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy],
                          'auc': [auc]
                        })
  
    return table


# Utilize the recently established function to extract all the scores derived from the grid search process. This will enable you to efficiently gather and analyze the comprehensive set of scores obtained during the search.

# In[56]:


# Get all CV scores
tree1_cv_results = make_results('decision tree cv', tree1, 'auc')
tree1_cv_results


# Each of the scores attained from the decision tree model serves as a robust indicator of favorable model performance.
# 
# It's important to bear in mind that decision trees have the potential for overfitting, where the model may capture noise in the data. In contrast, random forests mitigate overfitting by aggregating predictions from multiple trees. Consequently, constructing a random forest model is a logical progression to ensure enhanced predictive accuracy and generalization.

# #### Random forest

# Construct a random forest model and set up cross-validated grid-search to exhuastively search for the best model parameters.

# In[57]:


# Instantiate model
rf = RandomForestClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth': [3,5, None], 
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'n_estimators': [300, 500],
             }  

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
rf1 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')


# Fit the random forest model to the training data.

# In[58]:


get_ipython().run_cell_magic('time', '', 'rf1.fit(X_train, y_train) # --> Wall time: ~10min\n')


# Specify path to where you want to save your model.

# In[60]:


path = "C:\\Users\\37066\\OneDrive\\Desktop"


# Define functions to pickle the model and read in the model.

# In[61]:


def write_pickle(path, model_object, save_as:str):
    '''
    In: 
        path:         path of folder where you want to save the pickle
        model_object: a model you want to pickle
        save_as:      filename for how you want to save the model

    Out: A call to pickle the model in the folder indicated
    '''    

    with open(path + save_as + '.pickle', 'wb') as to_write:
        pickle.dump(model_object, to_write)


# In[62]:


def read_pickle(path, saved_model_name:str):
    '''
    In: 
        path:             path to folder where you want to read from
        saved_model_name: filename of pickled model you want to read in

    Out: 
        model: the pickled model 
    '''
    with open(path + saved_model_name + '.pickle', 'rb') as to_read:
        model = pickle.load(to_read)

    return model


# Use the functions defined above to save the model in a pickle file and then read it in.

# In[63]:


# Write pickle
write_pickle(path, rf1, 'hr_rf1')


# In[64]:


# Read pickle
rf1 = read_pickle(path, 'hr_rf1')


# Identify the best AUC score achieved by the random forest model on the training set.

# In[65]:


# Check best AUC score on CV
rf1.best_score_


# In[66]:


# Check best params
rf1.best_params_


# In[67]:


# Get all CV scores
rf1_cv_results = make_results('random forest cv', rf1, 'auc')
print(tree1_cv_results)
print(rf1_cv_results)


# The assessment metrics for the random forest model exhibit improvements over those of the decision tree model, showcasing a noteworthy enhancement in overall performance. The only marginal exception is observed in recall, where the random forest model registers a slightly lower score by approximately 0.001, an inconsequential difference. This substantial improvement highlights the predominance of the random forest model's efficacy.
# 
# Moving forward, the subsequent step involves evaluating the final model's performance on the test set, providing a comprehensive assessment of its predictive capabilities in a real-world context.

# Define a function that gets all the scores from a model's predictions.

# In[68]:


def get_scores(model_name:str, model, X_test_data, y_test_data):
    '''
    Generate a table of test scores.

    In: 
        model_name (string):  How you want your model to be named in the output table
        model:                A fit GridSearchCV object
        X_test_data:          numpy array of X_test data
        y_test_data:          numpy array of y_test data

    Out: pandas df of precision, recall, f1, accuracy, and AUC scores for your model
    '''

    preds = model.best_estimator_.predict(X_test_data)

    auc = roc_auc_score(y_test_data, preds)
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision], 
                          'recall': [recall],
                          'f1': [f1],
                          'accuracy': [accuracy],
                          'AUC': [auc]
                         })
  
    return table


# Now use the best performing model to predict on the test set.

# In[69]:


# Get predictions on test data
rf1_test_scores = get_scores('random forest1 test', rf1, X_test, y_test)
rf1_test_scores


# The test scores are very similar to the validation scores, which is good. This appears to be a strong model. Since this test set was only used for this model, you can be more confident that your model's performance on this data is representative of how it will perform on new, unseeen data.

# #### Feature Engineering

# Certainly, here's a rephrased version:
# 
# It's natural to be cautious about the high evaluation scores, as they could potentially indicate data leakage. Data leakage occurs when information that shouldn't be available during training inadvertently influences the model, leading to overinflated scores that may not hold true in real-world scenarios.
# 
# In this situation, it's likely that not all employees will have reported satisfaction levels, introducing a potential source of leakage. Furthermore, the `average_monthly_hours` column could also contribute to data leakage, especially if employees who are on the verge of leaving or being let go work fewer hours.
# 
# To address these concerns and enhance the model's reliability, the next phase involves refining the decision tree and random forest models through feature engineering. As a preliminary step, the `satisfaction_level` variable will be omitted. Additionally, a novel binary feature named `overworked` will be introduced, aimed at identifying whether an employee is working excessively.
# 
# This meticulous approach will bolster the model's robustness against data leakage, leading to more accurate and dependable performance evaluation.

# In[70]:


# Drop `satisfaction_level` and save resulting dataframe in new variable
df2 = df_enc.drop('satisfaction_level', axis=1)

# Display first few rows of new dataframe
df2.head()


# In[71]:


# Create `overworked` column. For now, it's identical to average monthly hours.
df2['overworked'] = df2['average_monthly_hours']

# Inspect max and min average monthly hours values
print('Max hours:', df2['overworked'].max())
print('Min hours:', df2['overworked'].min())


# Around 166.67 hours signifies the approximate monthly average for an individual working 8 hours a day, 5 days a week, for 50 weeks in a year.
# 
# Defining the notion of being overworked involves considering an average workload exceeding 175 hours per month.
# 
# For the purpose of creating a binary `overworked` column, you can effectively reassign values using a boolean mask:
# - `df3['overworked'] > 175` generates a boolean series, where `True` corresponds to values above 175 and `False` for values less than or equal to 175.
# - `.astype(int)` transforms `True` to `1` and `False` to `0`, effectively converting the series to binary form.

# In[72]:


# Define `overworked` as working > 175 hrs/week
df2['overworked'] = (df2['overworked'] > 175).astype(int)

# Display first few rows of new column
df2['overworked'].head()


# In[73]:


# Drop the `average_monthly_hours` column
df2 = df2.drop('average_monthly_hours', axis=1)

# Display first few rows of resulting dataframe
df2.head()


# Again, isolate the features and target variables

# In[75]:


# Isolate the outcome variable
y = df2['left']

# Select the features
X = df2.drop('left', axis=1)


# Split the data into training and testing sets.

# In[76]:


# Create test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


# #### Decision tree

# In[77]:


# Instantiate model
tree = DecisionTreeClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, 8, None],
             'min_samples_leaf': [2, 5, 1],
             'min_samples_split': [2, 4, 6]
             }

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
tree2 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')


# In[78]:


get_ipython().run_cell_magic('time', '', 'tree2.fit(X_train, y_train)\n')


# In[79]:


# Check best params
tree2.best_params_


# In[80]:


# Check best AUC score on CV
tree2.best_score_


# This model performs very well, even without satisfaction levels and detailed hours worked data. 
# 
# Next, check the other scores.

# In[82]:


# Get all CV scores
tree2_cv_results = make_results('decision tree2 cv', tree2, 'auc')
print(tree1_cv_results)
print(tree2_cv_results)


# As anticipated, a reduction in some of the other scores is not unexpected, considering the inclusion of fewer features in this iteration of the model. However, it's noteworthy that despite this constraint, the evaluation scores remain impressively high.

# #### Random forest II

# In[85]:


# Instantiate model
rf = RandomForestClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth': [3,5, None], 
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'n_estimators': [300, 500],
             }  

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
rf2 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')


# In[86]:


get_ipython().run_cell_magic('time', '', 'rf2.fit(X_train, y_train) # --> Wall time: 12mins 4s \n')


# In[87]:


# Write pickle
write_pickle(path, rf2, 'hr_rf2')


# In[88]:


# Read in pickle
rf2 = read_pickle(path, 'hr_rf2')


# In[89]:


# Check best params
rf2.best_params_


# In[90]:


# Check best AUC score on CV
rf2.best_score_


# In[91]:


# Get all CV scores
rf2_cv_results = make_results('random forest2 cv', rf2, 'auc')
print(tree2_cv_results)
print(rf2_cv_results)


# Once more, a minor decline in the scores is evident, albeit the random forest model continues to outperform the decision tree, particularly when assessing performance through the lens of AUC.
# 
# With this established, it's time to subject the champion model to scoring using the test set, thereby attaining a comprehensive evaluation of its predictive capabilities in a real-world context.

# In[92]:


# Get predictions on test data
rf2_test_scores = get_scores('random forest2 test', rf2, X_test, y_test)
rf2_test_scores


# This seems to be a stable, well-performing final model. 
# 
# Plot a confusion matrix to visualize how well it predicts on the test set.

# In[93]:


# Generate array of values for confusion matrix
preds = rf2.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=rf2.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=rf2.classes_)
disp.plot(values_format='');


# The model exhibits a tendency to predict more false positives than false negatives, implying instances where employees are flagged as potentially leaving or being dismissed, despite that not being the actual scenario. Nevertheless, it's crucial to acknowledge that despite this behavior, the model remains robust and reliable.
# 
# For further exploration, delving into the splits of the decision tree model and scrutinizing the most influential features within the random forest model could provide valuable insights into the inner workings of these models. This endeavor offers a deeper understanding of the decision-making processes and aids in uncovering the key factors driving the model's predictions.

# #### Decision tree splits

# In[94]:


# Plot the tree
plt.figure(figsize=(85,20))
plot_tree(tree2.best_estimator_, max_depth=6, fontsize=14, feature_names=X.columns, 
          class_names={0:'stayed', 1:'left'}, filled=True);
plt.show()


# It's worth noting that you have the option to double-click on the tree image, which will allow you to zoom in and meticulously examine the splits within the tree structure. This enables a detailed analysis of how the model arrives at its decisions at different branches of the tree.

# #### Insight into Feature Importance from Decision Trees
# 
# Moreover, extracting feature importance from decision trees can offer valuable insights (refer to the [DecisionTreeClassifier scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.feature_importances_) for comprehensive information). This analysis sheds light on the significance of individual features within the decision tree model, aiding in the identification of key contributors to the model's predictive capacity.

# In[95]:


#tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_, columns=X.columns)
tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_, 
                                 columns=['gini_importance'], 
                                 index=X.columns
                                )
tree2_importances = tree2_importances.sort_values(by='gini_importance', ascending=False)

# Only extract the features with importances > 0
tree2_importances = tree2_importances[tree2_importances['gini_importance'] != 0]
tree2_importances


# You can then create a barplot to visualize the decision tree feature importances.

# In[96]:


sns.barplot(data=tree2_importances, x="gini_importance", y=tree2_importances.index, orient='h')
plt.title("Decision Tree: Feature Importances for Employee Leaving", fontsize=12)
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.show()


# The presented barplot visually conveys the significance hierarchy within the decision tree model. Notably, the variables `last_evaluation`, `number_project`, `tenure`, and `overworked` emerge as the most influential factors, ordered by their respective importance scores. These specific variables play a pivotal role in predicting the outcome variable, `left`, underscoring their prominence in guiding the model's predictions.

# #### Random forest feature importance
# 
# Now, plot the feature importances for the random forest model.

# In[97]:


# Get feature importances
feat_impt = rf2.best_estimator_.feature_importances_

# Get indices of top 10 features
ind = np.argpartition(rf2.best_estimator_.feature_importances_, -10)[-10:]

# Get column labels of top 10 features 
feat = X.columns[ind]

# Filter `feat_impt` to consist of top 10 feature importances
feat_impt = feat_impt[ind]

y_df = pd.DataFrame({"Feature":feat,"Importance":feat_impt})
y_sort_df = y_df.sort_values("Importance")
fig = plt.figure()
ax1 = fig.add_subplot(111)

y_sort_df.plot(kind='barh',ax=ax1,x="Feature",y="Importance")

ax1.set_title("Random Forest: Feature Importances for Employee Leaving", fontsize=12)
ax1.set_ylabel("Feature")
ax1.set_xlabel("Importance")

plt.show()


# The depicted plot visually illustrates the hierarchy of importance within the random forest model. Notably, the variables `last_evaluation`, `number_project`, `tenure`, and `overworked` emerge as the foremost influential factors, ordered by their respective importance scores. These specific variables hold substantial predictive power for the outcome variable, `left`, and intriguingly, they align with the key variables utilized by the decision tree model.

# # pacE: Execute Stage
# - Interpret model performance and results
# - Share actionable steps with stakeholders
# 
# 

# 
# ## Recap of Evaluation Metrics
# 
# - **AUC (Area Under the ROC Curve)**: This metric quantifies the area beneath the Receiver Operating Characteristic (ROC) curve. It also represents the probability that the model ranks a randomly selected positive example higher than a randomly selected negative example.
# 
# - **Precision**: Precision calculates the ratio of true positive predictions to the total number of predicted positives. In simpler terms, it measures the accuracy of positive predictions.
# 
# - **Recall**: Recall calculates the ratio of true positive predictions to the total number of actual positives. It gauges the model's ability to correctly identify positive instances.
# 
# - **Accuracy**: Accuracy measures the ratio of correctly classified data points (both true positives and true negatives) to the total number of data points.
# 
# - **F1-Score**: The F1-score is a harmonic mean of precision and recall. It provides a balanced assessment of both metrics and is especially useful when there's an uneven class distribution.
# 
# These evaluation metrics offer comprehensive insights into the performance and effectiveness of a classification model, helping to assess its predictive capabilities from various perspectives.

# 💭 Reflection Questions:
# 
# 1. **Key Insights**: What significant findings or insights did you uncover through the model-building process? How did these insights contribute to a deeper understanding of the data and its implications?
# 
# 2. **Business Recommendations**: What actionable recommendations can be made based on the models' outcomes? How can the company leverage these insights to enhance employee retention and overall operations?
# 
# 3. **Manager/Company Recommendations**: If you were to provide recommendations to your manager or the company, what additional steps or initiatives would you propose based on the model's outcomes? How could these recommendations drive strategic decisions?
# 
# 4. **Model Improvement**: Do you believe your model has room for improvement? If so, why and how? Are there specific aspects of the data, features, or model architecture that you would address to enhance performance?
# 
# 5. **Further Questions**: Given your familiarity with the data and the models employed, what other questions could you explore to provide additional value to the team or organization? How might these questions address different aspects of employee engagement, satisfaction, or retention?
# 
# 6. **Resources Used**: What resources, tools, or references were crucial to your work during this stage? Please include any pertinent links that you found valuable.
# 
# 7. **Ethical Considerations**: Did you encounter any ethical considerations while handling the data or interpreting the results? How did you address or mitigate these concerns to ensure responsible and ethical data usage?
# 
# Reflecting on these questions will help you consolidate your understanding of the model's outcomes, assess its implications, and consider potential avenues for further analysis or improvement.

# ## Step 4. Results and Evaluation
# - Interpret model
# - Evaluate model performance using metrics
# - Prepare results, visualizations, and actionable steps to share with stakeholders
# 
# 
# 

# ### Summary of Model Results
# 
# **Logistic Regression**
# 
# The logistic regression model demonstrated commendable performance on the test set, with weighted average precision of 80%, recall of 83%, f1-score of 80%, and an accuracy level of 83%.
# 
# **Tree-based Machine Learning**
# 
# Following feature engineering, the decision tree model exhibited notable performance metrics on the test set: AUC of 93.8%, precision of 87.0%, recall of 90.4%, f1-score of 88.7%, and accuracy of 96.2%. Interestingly, the random forest model showcased a slightly superior performance compared to the decision tree model.
# 
# These outcomes provide valuable insights into the models' predictive capacities, highlighting their effectiveness in identifying and classifying employees at risk of leaving the company.

# ### Conclusion, Recommendations, and Next Steps
# 
# The comprehensive analysis conducted through the models, coupled with the derived feature importances, substantiates a significant observation: employees at the company are grappling with excessive workloads.
# 
# To foster employee retention and well-being, the following recommendations are proposed for consideration by stakeholders:
# 
# * **Project Cap**: Introduce a cap on the number of projects an employee can concurrently work on to mitigate overburdening.
# * **Four-Year Tenure Focus**: Investigate and address dissatisfaction among employees with a four-year tenure, potentially through targeted initiatives or policy adjustments.
# * **Balanced Rewards**: Equitably reward employees for extended work hours or reconsider expectations to prevent undue strain.
# * **Clarity in Policies**: Ensure clear communication of overtime pay policies and establish transparent expectations about workload and time off.
# * **Cultural Reflection**: Engage in company-wide and team-level discussions to comprehend and address underlying work culture concerns.
# * **Performance Scaling**: Revise the evaluation system to reflect proportional recognition for employees who contribute or exert more effort.
# 
# **Next Steps**
# 
# Continued vigilance regarding potential data leakage is warranted. Assessing model performance when `last_evaluation` is removed could provide valuable insights. Considering the feasibility of predicting employee retention without relying heavily on infrequent evaluations is also advisable. Moreover, examining the influence of evaluation and satisfaction scores on employee retention or even predicting performance scores could yield intriguing outcomes.
# 
# For a future initiative, the exploration of a K-means model and cluster analysis using this dataset offers an avenue to glean additional valuable insights into underlying patterns and group dynamics. This could provide a holistic understanding of employee engagement and retention drivers.
