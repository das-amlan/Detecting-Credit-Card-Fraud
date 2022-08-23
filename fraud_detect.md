Detecting Credit Card Fraud
================
Amlan Das
2022-08-23

------------------------------------------------------------------------

## Importing imporatnt libraries

``` r
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
library(ranger)
library(psych)
```

    ## 
    ## Attaching package: 'psych'

    ## The following objects are masked from 'package:ggplot2':
    ## 
    ##     %+%, alpha

``` r
library(data.table)
library(caTools)
library(pROC)
```

    ## Type 'citation("pROC")' for a citation.

    ## 
    ## Attaching package: 'pROC'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     cov, smooth, var

``` r
library(rpart)
library(rpart.plot)
library(gbm)
```

    ## Loaded gbm 2.1.8.1

## Importing the dataset

``` r
cc_data <- read.csv("creditcard.csv")
```

## Data Exploration

``` r
head(cc_data)
```

    ##   Time         V1          V2        V3         V4          V5          V6
    ## 1    0 -1.3598071 -0.07278117 2.5363467  1.3781552 -0.33832077  0.46238778
    ## 2    0  1.1918571  0.26615071 0.1664801  0.4481541  0.06001765 -0.08236081
    ## 3    1 -1.3583541 -1.34016307 1.7732093  0.3797796 -0.50319813  1.80049938
    ## 4    1 -0.9662717 -0.18522601 1.7929933 -0.8632913 -0.01030888  1.24720317
    ## 5    2 -1.1582331  0.87773675 1.5487178  0.4030339 -0.40719338  0.09592146
    ## 6    2 -0.4259659  0.96052304 1.1411093 -0.1682521  0.42098688 -0.02972755
    ##            V7          V8         V9         V10        V11         V12
    ## 1  0.23959855  0.09869790  0.3637870  0.09079417 -0.5515995 -0.61780086
    ## 2 -0.07880298  0.08510165 -0.2554251 -0.16697441  1.6127267  1.06523531
    ## 3  0.79146096  0.24767579 -1.5146543  0.20764287  0.6245015  0.06608369
    ## 4  0.23760894  0.37743587 -1.3870241 -0.05495192 -0.2264873  0.17822823
    ## 5  0.59294075 -0.27053268  0.8177393  0.75307443 -0.8228429  0.53819555
    ## 6  0.47620095  0.26031433 -0.5686714 -0.37140720  1.3412620  0.35989384
    ##          V13        V14        V15        V16         V17         V18
    ## 1 -0.9913898 -0.3111694  1.4681770 -0.4704005  0.20797124  0.02579058
    ## 2  0.4890950 -0.1437723  0.6355581  0.4639170 -0.11480466 -0.18336127
    ## 3  0.7172927 -0.1659459  2.3458649 -2.8900832  1.10996938 -0.12135931
    ## 4  0.5077569 -0.2879237 -0.6314181 -1.0596472 -0.68409279  1.96577500
    ## 5  1.3458516 -1.1196698  0.1751211 -0.4514492 -0.23703324 -0.03819479
    ## 6 -0.3580907 -0.1371337  0.5176168  0.4017259 -0.05813282  0.06865315
    ##           V19         V20          V21          V22         V23         V24
    ## 1  0.40399296  0.25141210 -0.018306778  0.277837576 -0.11047391  0.06692807
    ## 2 -0.14578304 -0.06908314 -0.225775248 -0.638671953  0.10128802 -0.33984648
    ## 3 -2.26185710  0.52497973  0.247998153  0.771679402  0.90941226 -0.68928096
    ## 4 -1.23262197 -0.20803778 -0.108300452  0.005273597 -0.19032052 -1.17557533
    ## 5  0.80348692  0.40854236 -0.009430697  0.798278495 -0.13745808  0.14126698
    ## 6 -0.03319379  0.08496767 -0.208253515 -0.559824796 -0.02639767 -0.37142658
    ##          V25        V26          V27         V28 Amount Class
    ## 1  0.1285394 -0.1891148  0.133558377 -0.02105305 149.62     0
    ## 2  0.1671704  0.1258945 -0.008983099  0.01472417   2.69     0
    ## 3 -0.3276418 -0.1390966 -0.055352794 -0.05975184 378.66     0
    ## 4  0.6473760 -0.2219288  0.062722849  0.06145763 123.50     0
    ## 5 -0.2060096  0.5022922  0.219422230  0.21515315  69.99     0
    ## 6 -0.2327938  0.1059148  0.253844225  0.08108026   3.67     0

``` r
dim(cc_data)
```

    ## [1] 284807     31

``` r
names(cc_data)
```

    ##  [1] "Time"   "V1"     "V2"     "V3"     "V4"     "V5"     "V6"     "V7"    
    ##  [9] "V8"     "V9"     "V10"    "V11"    "V12"    "V13"    "V14"    "V15"   
    ## [17] "V16"    "V17"    "V18"    "V19"    "V20"    "V21"    "V22"    "V23"   
    ## [25] "V24"    "V25"    "V26"    "V27"    "V28"    "Amount" "Class"

### Checking for missing values

We can see there is no missing value in our dataset

``` r
colSums(is.na(cc_data))
```

    ##   Time     V1     V2     V3     V4     V5     V6     V7     V8     V9    V10 
    ##      0      0      0      0      0      0      0      0      0      0      0 
    ##    V11    V12    V13    V14    V15    V16    V17    V18    V19    V20    V21 
    ##      0      0      0      0      0      0      0      0      0      0      0 
    ##    V22    V23    V24    V25    V26    V27    V28 Amount  Class 
    ##      0      0      0      0      0      0      0      0      0

``` r
describe(cc_data$Amount)
```

    ##    vars      n  mean     sd median trimmed   mad min      max    range  skew
    ## X1    1 284807 88.35 250.12     22   41.64 29.98   0 25691.16 25691.16 16.98
    ##    kurtosis   se
    ## X1   845.07 0.47

``` r
table(cc_data$Class)
```

    ## 
    ##      0      1 
    ## 284315    492

## Data Manipulation

-   Scaling the **“Amount”** column with respect to the dataset
-   Deleting **“Time”** column

``` r
cc_data$Amount <- scale(cc_data$Amount)
new_cc_data <- cc_data[,-c(1)] 
head(new_cc_data)
```

    ##           V1          V2        V3         V4          V5          V6
    ## 1 -1.3598071 -0.07278117 2.5363467  1.3781552 -0.33832077  0.46238778
    ## 2  1.1918571  0.26615071 0.1664801  0.4481541  0.06001765 -0.08236081
    ## 3 -1.3583541 -1.34016307 1.7732093  0.3797796 -0.50319813  1.80049938
    ## 4 -0.9662717 -0.18522601 1.7929933 -0.8632913 -0.01030888  1.24720317
    ## 5 -1.1582331  0.87773675 1.5487178  0.4030339 -0.40719338  0.09592146
    ## 6 -0.4259659  0.96052304 1.1411093 -0.1682521  0.42098688 -0.02972755
    ##            V7          V8         V9         V10        V11         V12
    ## 1  0.23959855  0.09869790  0.3637870  0.09079417 -0.5515995 -0.61780086
    ## 2 -0.07880298  0.08510165 -0.2554251 -0.16697441  1.6127267  1.06523531
    ## 3  0.79146096  0.24767579 -1.5146543  0.20764287  0.6245015  0.06608369
    ## 4  0.23760894  0.37743587 -1.3870241 -0.05495192 -0.2264873  0.17822823
    ## 5  0.59294075 -0.27053268  0.8177393  0.75307443 -0.8228429  0.53819555
    ## 6  0.47620095  0.26031433 -0.5686714 -0.37140720  1.3412620  0.35989384
    ##          V13        V14        V15        V16         V17         V18
    ## 1 -0.9913898 -0.3111694  1.4681770 -0.4704005  0.20797124  0.02579058
    ## 2  0.4890950 -0.1437723  0.6355581  0.4639170 -0.11480466 -0.18336127
    ## 3  0.7172927 -0.1659459  2.3458649 -2.8900832  1.10996938 -0.12135931
    ## 4  0.5077569 -0.2879237 -0.6314181 -1.0596472 -0.68409279  1.96577500
    ## 5  1.3458516 -1.1196698  0.1751211 -0.4514492 -0.23703324 -0.03819479
    ## 6 -0.3580907 -0.1371337  0.5176168  0.4017259 -0.05813282  0.06865315
    ##           V19         V20          V21          V22         V23         V24
    ## 1  0.40399296  0.25141210 -0.018306778  0.277837576 -0.11047391  0.06692807
    ## 2 -0.14578304 -0.06908314 -0.225775248 -0.638671953  0.10128802 -0.33984648
    ## 3 -2.26185710  0.52497973  0.247998153  0.771679402  0.90941226 -0.68928096
    ## 4 -1.23262197 -0.20803778 -0.108300452  0.005273597 -0.19032052 -1.17557533
    ## 5  0.80348692  0.40854236 -0.009430697  0.798278495 -0.13745808  0.14126698
    ## 6 -0.03319379  0.08496767 -0.208253515 -0.559824796 -0.02639767 -0.37142658
    ##          V25        V26          V27         V28      Amount Class
    ## 1  0.1285394 -0.1891148  0.133558377 -0.02105305  0.24496383     0
    ## 2  0.1671704  0.1258945 -0.008983099  0.01472417 -0.34247394     0
    ## 3 -0.3276418 -0.1390966 -0.055352794 -0.05975184  1.16068389     0
    ## 4  0.6473760 -0.2219288  0.062722849  0.06145763  0.14053401     0
    ## 5 -0.2060096  0.5022922  0.219422230  0.21515315 -0.07340321     0
    ## 6 -0.2327938  0.1059148  0.253844225  0.08108026 -0.33855582     0

## Data Modeling

### Split our dataset into **Training Set** and **Test Set**

``` r
set.seed(123)
split_data <- sample.split(new_cc_data$Class,SplitRatio=0.80)
train_data <- subset(new_cc_data,split_data==TRUE)
test_data <- subset(new_cc_data,split_data ==FALSE)
dim(train_data)
```

    ## [1] 227846     30

``` r
dim(test_data)
```

    ## [1] 56961    30

### Fitting ML Model

#### Logistic Regression Model

``` r
lr_model <- glm(Class~., 
                train_data, 
                family = binomial())
```

``` r
summary(lr_model)
```

    ## 
    ## Call:
    ## glm(formula = Class ~ ., family = binomial(), data = train_data)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -4.6108  -0.0292  -0.0194  -0.0125   4.6021  
    ## 
    ## Coefficients:
    ##              Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -8.651305   0.160212 -53.999  < 2e-16 ***
    ## V1           0.072540   0.044144   1.643 0.100332    
    ## V2           0.014818   0.059777   0.248 0.804220    
    ## V3           0.026109   0.049776   0.525 0.599906    
    ## V4           0.681286   0.078071   8.726  < 2e-16 ***
    ## V5           0.087938   0.071553   1.229 0.219079    
    ## V6          -0.148083   0.085192  -1.738 0.082170 .  
    ## V7          -0.117344   0.068940  -1.702 0.088731 .  
    ## V8          -0.146045   0.035667  -4.095 4.23e-05 ***
    ## V9          -0.339828   0.117595  -2.890 0.003855 ** 
    ## V10         -0.785462   0.098486  -7.975 1.52e-15 ***
    ## V11          0.001492   0.085147   0.018 0.986018    
    ## V12          0.087106   0.094869   0.918 0.358532    
    ## V13         -0.343792   0.092381  -3.721 0.000198 ***
    ## V14         -0.526828   0.067084  -7.853 4.05e-15 ***
    ## V15         -0.095471   0.094037  -1.015 0.309991    
    ## V16         -0.130225   0.138629  -0.939 0.347537    
    ## V17          0.032463   0.074471   0.436 0.662900    
    ## V18         -0.100964   0.140985  -0.716 0.473909    
    ## V19          0.083711   0.105134   0.796 0.425897    
    ## V20         -0.463946   0.081871  -5.667 1.46e-08 ***
    ## V21          0.381206   0.065880   5.786 7.19e-09 ***
    ## V22          0.610874   0.142086   4.299 1.71e-05 ***
    ## V23         -0.071406   0.058799  -1.214 0.224589    
    ## V24          0.255791   0.170568   1.500 0.133706    
    ## V25         -0.073955   0.142634  -0.519 0.604109    
    ## V26          0.120841   0.202553   0.597 0.550783    
    ## V27         -0.852018   0.118391  -7.197 6.17e-13 ***
    ## V28         -0.323854   0.090075  -3.595 0.000324 ***
    ## Amount       0.292477   0.092075   3.177 0.001491 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 5799.1  on 227845  degrees of freedom
    ## Residual deviance: 1790.9  on 227816  degrees of freedom
    ## AIC: 1850.9
    ## 
    ## Number of Fisher Scoring iterations: 12

#### ROC Curve

``` r
lr_predict <- predict(lr_model, 
                      test_data, 
                      probability = TRUE)

auc_lr <- roc(test_data$Class, 
              lr_predict, 
              plot = TRUE, 
              col = "blue")
```

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

![](fraud_detect_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

``` r
print(auc_lr)
```

    ## 
    ## Call:
    ## roc.default(response = test_data$Class, predictor = lr_predict,     plot = TRUE, col = "blue")
    ## 
    ## Data: lr_predict in 56863 controls (test_data$Class 0) < 98 cases (test_data$Class 1).
    ## Area under the curve: 0.9748

We can see that we get a accuracy of 97% from logistic regression model.
But the output is expected as we have a very imbalanced data. So even if
logistic regression is giving a good result it is not reliable.

#### Decision Tree Model

``` r
decisionTree_model <- rpart(Class ~ . , 
                            new_cc_data, 
                            method = 'class')
predicted_dt <- predict(decisionTree_model, 
                         new_cc_data, 
                         type = 'class')
probability <- predict(decisionTree_model, 
                       new_cc_data, 
                       type = 'prob')
rpart.plot(decisionTree_model)
```

![](fraud_detect_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

#### Gradient Boosting (GBM)

``` r
system.time(
         model_gbm <- gbm(Class ~ .
               , distribution = "bernoulli"
               , data = rbind(train_data, test_data)
               , n.trees = 500
               , interaction.depth = 3
               , n.minobsinnode = 100
               , shrinkage = 0.01
               , bag.fraction = 0.5
               , train.fraction = nrow(train_data) / (nrow(train_data) + nrow(test_data))
)
)
```

    ##    user  system elapsed 
    ##  448.46    0.71  456.54

``` r
#Determine best iteration based on test data
gbm.iter <- gbm.perf(model_gbm, 
                    method = "test")
```

![](fraud_detect_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

#### ROC Curve

``` r
gbm_test <- predict(model_gbm, 
                   newdata = test_data, 
                   n.trees = gbm.iter)
gbm_auc <- roc(test_data$Class, 
              gbm_test, 
              plot = TRUE, 
              col = "red")
```

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

![](fraud_detect_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

``` r
print(gbm_auc)
```

    ## 
    ## Call:
    ## roc.default(response = test_data$Class, predictor = gbm_test,     plot = TRUE, col = "red")
    ## 
    ## Data: gbm_test in 56863 controls (test_data$Class 0) < 98 cases (test_data$Class 1).
    ## Area under the curve: 0.9541
