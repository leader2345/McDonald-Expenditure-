# McDonald-Expenditure-

## Introduction

Using my own expenditure dataset that I've tracked over the last 6 years, I'm going to use it to predict how much I'll spend on McDonald based on the attributes: Food Category and Month in the dataset. The mobile app that I've used is MoneyLover App: https://moneylover.me/.

I'll start by loading the dataset:
![Dataset](Blog Pictures\Loading_the_data_code)

![Dataset_Table](Blog Pictures\Loading_the_data_table)

I'll be only taking the important attributes such as: ID, Notes, Amount, Category and the Date for this project.

As seen in the image below, there are 2 missing values in Notes out of 4983 non-null values.
Let's take a look at the where the null values are in the Notes category:

![ExpDataset](Blog_Pictures\Data_Info)
![ExpDataset table](Blog_Pictures\Checking_for_null_values_table)
![Null values](Blog_Pictures\Checking_for_null_values_table)

It looks like the null values are under the Driving Lessons and Salary Category.
As we are only getting the Food Category, we can ignore these as they'll be dropped.

## Getting the important attributes
I'll be getting only the most important Attributes:
* Notes
* Amount
* Category
* Date

Code below:
```
exp_data_1 = pd.DataFrame(exp_dataset, columns=["Notes", "Amount", "Category", "Date"])
exp_data_1.head()
```
![Impt attributes](Blog_Pictures\Getting_impt_attributes)

I can



Numerical Attribute:
* Amount

Categorical Attributes:
* Day, Months, Quarterly Period, Year


Predictors (X-axis)
Group by:
* Months
* ["Jan", "Feb", "Mar", "Apr", ...]

Food Category
* ["Beef", "Chicken", "Desserts", "Fish", "Sausage"]

Labels (Y-axis)
* Amount
