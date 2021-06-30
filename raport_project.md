EDA

1.  Data quality is good, datasets don’t include any null values, there
    are also just a few outliers, we decided to don’t delete any row or
    column.

2.  To check descriptive statistics we used pandas.describe and
    pandas.var.

3.  Next, we decided to check the distribution of the label's value. We
    noticed that there is 10 times more “1” than “-1”. It means that the
    dataset is unbalanced.

4.  Data after visualization looked like that:

![](https://i.imgur.com/poWhOEM.png)

Metric

1.  We made a mistake at first, we chose accuracy, but it was not a good
    metric, because with unbalanced data we got high accuracy, because
    model just predict the most common class. When we saw it, we decided
    to use F1 score. It is much better choice in this case because it is
    a connection between 2 important metrics, precision and recall.

        The formula for the F1 score is:

     

      F1 = 2 \* (precision \* recall) / (precision + recall)

Baseline

 

We decided to use
[sklearn.dummy](https://www.google.com/url?q=https://scikit-learn.org/stable/modules/classes.html%23module-sklearn.dummy&sa=D&source=editors&ust=1625095867936000&usg=AOvVaw3EwzpY9z4len9Kq4301mUd).DummyClassifier
as our baseline, the score we got is:

| Labels:                              | F1-score                             |
|--------------------------------------|--------------------------------------|
| -1                                   | 0.09                                 |
| 1                                    | 0.90                                 |

As we can see, the model can’t handle minor class, we see this problem,
as we chose a correct metric. Baseline score is unacceptable, so we need
to improve it.

Dataset split

 

To split dataset we decided to use
sklearn.model\_selection.train\_test\_split with params:

test\_size=0.33,

random\_state=42,

stratify=y

GridSearch

To find the best params and classifier, we decided to use Gridsearch. We
got those params:

 XGBClassifier(base\_score=0.5, booster='gbtree', colsample\_bylevel=1,

              colsample\_bynode=1, colsample\_bytree=1, gamma=0,

              learning\_rate=0.1, max\_delta\_step=0, max\_depth=3,

              min\_child\_weight=1, missing=None, n\_estimators=50,
n\_jobs=1,

              nthread=None, objective='binary:logistic',
random\_state=0,

              reg\_alpha=0, reg\_lambda=1, scale\_pos\_weight=1,
seed=None,

              silent=None, subsample=1, verbosity=1)

Conclusions

| Baseline           | F1 Score           | Best classifier    | F1 Score           |
|--------------------|--------------------|--------------------|--------------------|
| -1                 | 0.09               | -1                 | 0.88               |
| 1                  | 0.90               | 1                  | 0.99               |

               

Confusion matrix baseline
|---|                 positive        |                negative           |
|---|--------------------------------------|--------------------------------------|
|positive| 10                                   | 114                                  |
|negative| 118                                  | 996                                  |


Confusion matrix best classifier 
|---|                 positive        |                negative           |
|---|--------------------------------------|--------------------------------------|
|positive| 111                                  | 13                                   |
|negative| 18                                   | 1096                                 |

             

Classification raport baseline:

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| -1           | 0.10      | 0.09   | 0.09     | 124     |
| 1            | 0.90      | 0.91   | 0.90     | 1114    |
| accuracy     |           |        | 0.83     | 1238    |
| macro avg    | 0.50      | 0.50   | 0.50     | 1238    |
| weighted avg | 0.82      | 0.83   | 0.82     | 1238    |

Classification raport best classifier:

| precision    | recall | f1-score | support |      |
|--------------|--------|----------|---------|------|
| -1           | 0.86   | 0.90     | 0.88    | 124  |
| 1            | 0.99   | 0.98     | 0.99    | 1114 |
| accuracy     |        |          | 0.97    | 1238 |
| macro avg    | 0.92   | 0.94     | 0.93    | 1238 |
| weighted avg | 0.98   | 0.97     | 0.98    | 1238 |