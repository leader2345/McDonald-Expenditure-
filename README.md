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

<img src = "Blog Pictures/Null_values_notes_table.png" alt="">
<br></br>

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

<img src = "Blog Pictures/exp_data_1_head_10.png" alt="">
<br></br>

The **Amount** Category displays the (-) sign. It needs to be converted to positive. I'll be using the abs()
function to convert it to positive integers.

```
exp_data_1["Amount"]= exp_data_1["Amount"].abs()
exp_data_1.head()
```
<img src="Blog Pictures/Converting_to_abs_values.png">

## Getting the Date Time
I'm going to get the dates and have them split into:
* Days
* Month
* Quarterly Periods
* Years

Then, I'll insert them back into the dataframe and drop the "Date" attribute.

First, I'll get the Date attribute.

```
exp_data_1["Date"]
```

```
ID
4995    16/05/2020
4994    16/05/2020
4993    16/05/2020
4992    16/05/2020
4991    16/05/2020
           ...    
3540    12/04/2014
3550    10/04/2014
3549    10/04/2014
3548    10/04/2014
3546    10/04/2014
Name: Date, Length: 3631, dtype: object
```

Notice that the data type is an *object*. It needs to be converted to a *datetime64[ns]* format.

```
All_Date = pd.to_datetime(exp_data_1["Date"], dayfirst=True)
All_Date
```

```
ID
4995   2020-05-16
4994   2020-05-16
4993   2020-05-16
4992   2020-05-16
4991   2020-05-16
          ...    
3540   2014-04-12
3550   2014-04-10
3549   2014-04-10
3548   2014-04-10
3546   2014-04-10
Name: Date, Length: 3631, dtype: datetime64[ns]
```
Now, that it's converted, I can proceed with creating the Categorical Attributes: Days, Months, Quarterly Period and Years, for the "Date" Attribute.

Then, I'll add all these attributes back into the original dataset and drop the "Date" column.

## Splitting into the Quarterly, Monthly and Daily Period Columns

### Quarterly Period
To convert the Dates to quarterly periods, first I've to make a copy of the *exp_dataset_1*, this is to prevent making changes to the original dataset. Then, slice the "Date" column and convert it to Quarterly Periods.

```
exp_data_1_date = exp_data_1.copy()
exp_date_quarterly = exp_data_1_date["Date"].dt.to_period("Q")
exp_date_quarterly
```

```
ID
4995    2020Q2
4994    2020Q2
4993    2020Q2
4992    2020Q2
4991    2020Q2
         ...  
3540    2014Q2
3550    2014Q2
3549    2014Q2
3548    2014Q2
3546    2014Q2
Name: Date, Length: 3631, dtype: period[Q-DEC]
```
## Monthly
Similarly, I'll obtain the "Date" column and convert to months in their integer format.
```
exp_date_monthly = exp_data_1_date["Date"].dt.month
np.sort(pd.unique(exp_date_monthly))
```

```
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], dtype=int64)
```

```
exp_date_monthly
```

```
ID
4995    5
4994    5
4993    5
4992    5
4991    5
       ..
3540    4
3550    4
3549    4
3548    4
3546    4
Name: Date, Length: 3631, dtype: int64
```

## Day of the week
I'll convert the "Date" to "Days" their integer format.

```
exp_date_days_1 = exp_data_1_date["Date"].dt.dayofweek+1
exp_date_days_1
```

```
ID
4995    6
4994    6
4993    6
4992    6
4991    6
       ..
3540    6
3550    4
3549    4
3548    4
3546    4
Name: Date, Length: 3631, dtype: int64
```

```
np.sort(pd.unique(exp_date_days_1))
```

```
array([1, 2, 3, 4, 5, 6, 7], dtype=int64)
```
<br>I've added +1 as the first day starts with 0. The integers corresponding to their day of the week are seen below.</br>

Mon: 1

Tue: 2

Wed: 3

Thu: 4

Fri: 5

Sat: 6

Sun: 7

## Year Column
Creating the "Year" column.
```
exp_date_yearly =exp_data_1_date["Date"].dt.year
exp_date_yearly
```

```
ID
4995    2020
4994    2020
4993    2020
4992    2020
4991    2020
        ...
3540    2014
3550    2014
3549    2014
3548    2014
3546    2014
Name: Date, Length: 3631, dtype: int64
```

# Insert the Days, Month, Quarterly Period and Year columns into the original dataset

```
exp_data_1.insert(3, "Day", exp_date_days_1)
exp_date_1.insert(4, "Month", exp_date_monthly)
exp_data_1.insert(5, "Quarterly Period", exp_date_quarterly)
exp_date_1.insert(6, "Year", exp_date_yearly)
exp_data_1
```

<img src="Blog Pictures/Exp_data_1_new_columns.png">
<br></br>

Next, we move on to creating the different food categories under the "Category" column

# Getting the McDonald's Food from Category column
Now, I've to obtain all of the McDonald Food from the "Notes" column.

First, I've identified that all McDonald's Food start with the string "McDonald" and thus, we obtain only the "Notes" with "McDonald" string in them. Also, I've specified the string to be non-case sensitive and disable regular expression

```
McDonald_Exp = exp_data_1[exp_data_1["Notes"].str.contains("McDonald", case=False, regex=False)]
McDonald_Exp.sort_index()
```
<img src="Blog Pictures/McDonald_Exp.png"/>
<br></br>

# Splitting the Category column into 5 different Categories

Now, I'll move on to Category column to split the "Food & Beverage" into 5 different categories:

* Fish
* Chicken
* Sausage
* Beef
* Dessert

## Obtaining the Fish category


```
McDonald_FishBurger = McDonald_Exp[McDonald_Exp["Notes"].str.contains("Fish"), case=False].copy()
McDonald_FishBurger["Notes"].value_counts()
```
```
McDonald's Filet O Fish Burger                          28
McDonald's Double Filet O Fish                          10
McDonald's Double Filet O Fish meal                      3
Sentosa McDonald's double file o fish                    1
Mcdonald filet fish Burger                               1
McDonald's Filet O Fish meal with criss cut fries        1
Mcdonald double fillet o fish meal                       1
McDonald's sweet chili fish burger                       1
McDonald's double Nacho fillet fish                      1
McDonald's Chili Lime Fish Burger                        1
McDonald's Double Filet O Fish meal with curly fries     1
McDonald's nacho fillet fish                             1
McDonald's Filet O Fish meal                             1
Name: Notes, dtype: int64
```
I've added the .copy() as it'll throw warning message about attempting to replace a string in a copy of a slice in the DataFrame.

Next, we'll replace the "Food & Beverage" string with "Fish"
```
McDonald_FishBurger.loc[:, "Category"] = "Fish"
McDonald_FishBurger.head()
```

**Show Image here**

<br>I'll do the same for the rest of the Food Category.</br>

## Obtaining the Chicken Burger category
```
McDonald_ChickenBurger = McDonald_Exp[McDonald_Exp["Notes"].str.contains("Chicken|nuggets|Mcnugget|mcwings|mcspicy|nasi lemak|Ha Ha",case=False, regex=False)].copy()
McDonald_ChickenBurger.head()
```

**Insert Image here**

Replace "Food & Beverage" string with "Chicken"

```
McDonald_ChickenBurger.loc[:, "Category"] == "Chicken"
McDonald_ChickenBurger.head()
```

I'm going to be dropping *herb chicken pie* from as I want it to fall under desserts.
Later on, they're are going to be some overlapping of the Categories which will create duplicate rows, which is why I dropped it. I'll explain it later when I remove the duplicates.


```
herb_chicken_pie = McDonald_ChickenBurger[McDonald_ChickenBurger["Notes"].str.contains("herb chicken", case=False)]
herb_chicken_pie
```

**Insert Picture here**

```
McDonald_ChickenBurger = McDonald_ChickenBurger.drop(herb_chicken_pie.index)
```

Please refer to the jupyter notebook for the remaining **Food Categories** as it's is the same as the above

# Concatenating the Food Categories all together

I'll concantenate all of the Food Categories.

```
Frames = [McDonald_FishBurger, McDonald_ChickenBurger, McDonald_Sausage, McDonald_Beef, McDonald_Desserts]
```

```
Combined_Food_Cat = pd.concat(Frames)
Combined_Food_Cat.sort_values(by="ID", ascending=False)
```

**Insert Picture**

```
Combined_Food_Cat["Category"].value_counts()
```

```
Chicken     104
Fish         51
Beef         32
Sausage      23
Desserts     20
Name: Category, dtype: int64
```

## Check if there are any "Food & Beverage" in Combined_Food_Cat

```
Combined_Food_Cat[Combined_Food_Cat["Category"] == "Food & Beverage"]
```

There aren't any.

## Checking for duplicate values in Combined_Food_Cat

As the ID, which is the index of the DataFrame, is the unique value for each food, I'll be using it to check for the duplicate values for the food.

```
Combined_Food_Cat[Combined_Food_Cat.index.duplicated(keep=False)].sort_index()
```

**Insert Picture**

As mentioned earlier, the some of the food have their categories overlapped as some rows have 2 categories inside them. For simplicity, I'll be removing every second row and then append them back into the Combined_Food_Cat.






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
