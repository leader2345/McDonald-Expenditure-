# McDonald-Expenditure-

## Introduction

Using my own expenditure dataset that I've tracked over the last 6 years, I'm going to use it to predict how much I'll spend on McDonald based on the attributes: **Food Category** and **Month** in the dataset. The mobile app that I've used to track my expenditure is MoneyLover App: https://moneylover.me/.

I'll start by loading the dataset:

<img src = "Blog Pictures/Loading_the_data_code.png" alt="">

<img src = "Blog Pictures/Loading_the_data_table.png" alt="">

I'll be only taking the important attributes such as: ID, Notes, Amount, Category and the Date for this project.

As seen in the image below, there are 2 missing values in Notes out of 4983 non-null values.
Let's take a look at the what the null values are in the Notes category:

```
exp_dataset.info()

```
```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 4985 entries, 4996 to 3551
Data columns (total 8 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   Notes           4983 non-null   object
 1   Amount          4985 non-null   float64
 2   Category        4985 non-null   object
 3   Account         4985 non-null   object
 4   Currency        4985 non-null   object
 5   Date            4985 non-null   object
 6   Event           4985 non-null   object
 7   Exclude Report  0 non-null      float64
dtypes: float64(2), object(6)
memory usage: 350.5+ KB

```

```
print("Number of null values for each column")
pd.DataFrame(exp_dataset.isnull().sum(), columns=["Null values"])
```
```
Null values
Notes 	2
Amount 	0
Category 	0
Account 	0
Currency 	0
Date 	0
Event 	0
Exclude Report 	4985
```

```
exp_dataset[exp_dataset["Notes"].isnull().values]
```

<img src = "Blog Pictures/Null_values_notes_table.png" alt=""></img>

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

<img src = "Blog Pictures/Getting_impt_attributes.png" alt=""></img>


I'll get the number of rows for the top 5 Categories and display them.
```
exp_data_1["Category"].value_counts()[:5]

Food & Beverage    3631
Investment          529
Gifts               199
Transportation       95
Entertainment        87
Name: Category, dtype: int64
```

It looks like Food & Beverage have the highest number of instances.

## Getting only the Food & Beverage Category
I'll only be getting the Food & Beverage Category. The following code extracts the **Food & Beverage** category.

    F_and_B = exp_data_1["Category"] == "Food & Beverage"
    exp_data_1 = exp_data_1[F_and_B]
    exp_data_1.head(10)

<img src = "Blog Pictures/exp_data_1_head_10.png" alt=""></img>

The **Amount** Category displays the (-) sign. It needs to be converted to positive. I'll be using the abs()
function to convert it to positive integers.

```
exp_data_1["Amount"]= exp_data_1["Amount"].abs()
exp_data_1.head()
```
<img src="Blog Pictures/Converting_to_abs_values.png"></img>

## Getting the Date Time




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
