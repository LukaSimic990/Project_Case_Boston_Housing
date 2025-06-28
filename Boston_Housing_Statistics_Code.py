import pandas as pd
import requests as rq
import io 
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.stats as sst
import statsmodels.api as sm
from statsmodels.formula.api import ols

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_url = rq.get(url).text
data = io.StringIO(boston_url)
boston_df = pd.read_csv(data)


#Boxplot to show the median value of owner-occupied homes in $1000's in Boston
MEDV_median = boston_df['MEDV'].median()
ax = sb.boxplot(y='MEDV', data=boston_df)
ax.set_title("Chart showing the median value of owner-occupied homes")
ax.set_xlabel(f"Median value: {MEDV_median}")
ax.set_ylabel("Value of owner-occupied homes in 1000's")
plt.show()

#Bar plot of the River Charles variable (number of properties bounded/not bounded by River Charles)
boston_df.loc[(boston_df['CHAS']==1), 'Bounded_by_River_Charles']='Yes'
boston_df.loc[(boston_df['CHAS']==0), 'Bounded_by_River_Charles']='No'
sorted_counts = boston_df.Bounded_by_River_Charles.value_counts().sort_index()
plt.bar(sorted_counts.index, sorted_counts.values, color=['red', 'blue'])
plt.title('Bar plot of River Charles Variable')
plt.xlabel('Bounded by River Charles')
plt.ylabel('Count (number of properties)')
plt.show()

#Boxplot of MEDV vs AGE

#Discretizing of AGE varible
thirtyfive_lower = boston_df.loc[(boston_df['AGE']<=35), 'age_group']='35 and younger'
thirtyfive_seventy = boston_df.loc[(boston_df['AGE']>35)&(boston_df['AGE']<70), 'age_group']='between 35 and 70'
over_seventy = boston_df.loc[(boston_df['AGE']>=70), 'age_group']='over 70'

#Making the boxplot 
ax = sb.boxplot(x='age_group', y='MEDV', data=boston_df)
ax.set_title('MEDV vs AGE')
ax.set_xlabel('Age groups')
ax.set_ylabel("Median value of owner-occupied homes in $1000's")
plt.show()

#Scatterplot to show the relationship between nitric oxide concentrations and the 
#proportion of non-retail business acres per town 
ax = sb.scatterplot(x='NOX', y='INDUS', data=boston_df)
ax.set_title('NOX vs INDUS')
ax.set_xlabel('NOX-variable: Nitric oxides concentration (parts per 10 million)')
ax.set_ylabel('INDUS-variable: Proportion of non-retail business acres per town')
plt.show()

#Pupil to teacher ratio by town- histogram
plt.hist(boston_df['PTRATIO'])
plt.title('PTRATIO-variable: Pupil to teacher ratio by town')
plt.show()

#Is there a significant difference in median value of houses bounded by the Charles river or not?
print(sst.levene(boston_df[boston_df['Bounded_by_River_Charles']=='Yes']['MEDV'],
                 boston_df[boston_df['Bounded_by_River_Charles']=='No']['MEDV']))

print(sst.ttest_ind(boston_df[boston_df['Bounded_by_River_Charles']=='Yes']['MEDV'],
                    boston_df[boston_df['Bounded_by_River_Charles']=='No']['MEDV']))

#Is there a difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)? (ANOVA)
thirtyfive_lower = boston_df[boston_df['age_group']=='35 and younger']['MEDV']
thirtyfive_seventy = boston_df[boston_df['age_group']=='between 35 and 70']['MEDV']
over_seventy = boston_df[boston_df['age_group']=='over 70']['MEDV']

print(sst.levene(thirtyfive_lower, thirtyfive_seventy, over_seventy, center='median'))

f_statistic, p_value = sst.f_oneway(thirtyfive_lower, thirtyfive_seventy, over_seventy)
print("F-Statistic: {}; P-Value: {}".format(f_statistic, p_value))

#Can we conclude that there is no relationship between Nitric oxide concentrations 
#and proportion of non-retail business acres per town? (Pearson Correlation)(NOX, INDUS)
ax = sb.scatterplot(x='NOX', y='INDUS', data=boston_df)
ax.set_title('Correlation between nitric oxide and proportion of non-retail bussines acres per town')
ax.set_xlabel('Concentration of nitric oxide')
ax.set_ylabel('Proportion of non-retail business')
plt.show()

print(sst.pearsonr(boston_df['NOX'], boston_df['INDUS']))


#What is the impact of an additional weighted distance to the five Boston employment centres 
#on the median value of owner occupied homes? (Regression analysis)

#Scatterplot of dependance of median value of owner occupied homes (MEDV) on additional weighted distance 
#to the five Boston employment centers (DIS)
ax = sb.scatterplot(x='DIS', y='MEDV', data=boston_df)
ax.set_title('Dependance of MEDV on DIS')
ax.set_xlabel('DIS')
ax.set_ylabel('MEDV')
plt.show()

#First running pearsonr
print(sst.pearsonr(boston_df['DIS'], boston_df['MEDV']))

#Running the regression analysis
X = boston_df['DIS'] #Independent variable 
y = boston_df['MEDV'] #Dependent variable
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
predicitions = model.predict(X)
print(model.summary())



