# Language Detection — Model Comparison

| Model | Test Accuracy |
|---|---|
| Logistic Regression *(primary)* | **98.83%** |
| Linear SVC | 99.01% |
| Multinomial Naive Bayes | 98.14% |

## LogReg — Per-class Report

```
              precision    recall  f1-score   support

         ara       1.00      1.00      1.00       500
         bul       0.99      0.97      0.98       500
         ces       1.00      0.99      1.00       500
         cmn       0.98      0.99      0.99       500
         deu       0.96      0.98      0.97       500
         ell       1.00      0.99      0.99       500
         eng       0.91      1.00      0.95       500
         fra       0.98      0.99      0.99       500
         hin       1.00      0.98      0.99       500
         ita       0.99      0.99      0.99       500
         jpn       1.00      0.99      0.99       500
         kor       1.00      0.99      0.99       500
         nld       1.00      0.99      1.00       500
         pol       0.99      0.99      0.99       500
         por       0.99      0.99      0.99       500
         ron       1.00      0.99      0.99       500
         rus       0.99      1.00      0.99       500
         spa       0.99      0.98      0.98       500
         tur       1.00      0.99      1.00       500
         ukr       1.00      0.98      0.99       500

    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000

```
