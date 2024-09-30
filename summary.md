# Linear Regression

Linear regression is a simple approach to supervised learning. It assumes that the dependence of Y on $X_1, X_2, ..., X_p$ is linear. 
- True regression functions are never linear!

![img.png](figs/1/img.png)

- Although it may seem overly simplistic, linear regression is extremely useful both conceptually and practically.

## Linear regression for the advertising data

Consider the advertising data shown.

Questions we might ask:

- Is there a relationship between advertising budget and sales?
- How strong is the relationship between advertising budget and sales?
- Which media contribute to sales?
- How accurately can we predict future sales?
- Is the relationship linear?
- Is there synergy among the advertising media?

![img_1.png](figs/1/img_1.png)

## Simple linear regression using a single predictor X.

We assume a model

$$
Y = \beta_0 + \beta_1 X + \epsilon 
$$

where $\beta_0$ and $\beta_1$ are two unknown constants that represent the intercept and slope, also known as coefficients or parameters, and $\epsilon$ is the error term.

Given some estimates $\hat{\beta_0}$ and $\hat{\beta_1}$ for the model coefficients, we predict future sales using

$$
\hat{y} = \hat{\beta_0} + \hat{\beta_1}x, 
$$

where $\hat{y}$ indicates a prediction of Y on the basis of $X = x$. The hat symbol denotes an estimated value.

### Estimation of the parameters by least squares

Let $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$ be the prediction for $Y$ based on the $i$th value of $X$. Then $e_i = y_i - \hat{y}_i$ represents the $i$th residual.

We define the residuals sum of squares (RSS) as

$$ 
RSS = e_1^2 + e_2^2 + ... + e_n^2, 
$$

or equivalently as

$$ 
RSS = (y_1 - \hat{\beta}_0 - \hat{\beta}_1 x_1)^2 + (y_2 - \hat{\beta}_0 - \hat{\beta}_1 x_2)^2 + ... + (y_n - \hat{\beta}_0 - \hat{\beta}_1 x_n)^2. 
$$


The least squares approach chooses $\hat{\beta}_0$ and $\hat{\beta}_1$ to minimize the RSS. The minimizing values can be shown to be
$$
\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2},
$$

$$
\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x},
$$
where $\bar{y} = \frac{1}{n} \sum_{i=1} y_i$ and $\bar{x} = \frac{1}{n} \sum_{i=1} x_i$ are the sample means.

### Example: advertising data

![img_2.png](figs/1/img_2.png)

For the Advertising data, the least squares fit for the regression of sales onto TV is shown. The fit is found by minimizing the sum of squared errors. Each grey line segment represents an error, and the fit makes a compromise by averaging their squares. In this case, a linear fit captures the essence of the relationship, although it is somewhat deficient in the left of the plot.

### Assessing the Accuracy of the Coefficient Estimates
The standard error of an estimator reflects how it varies under repeated sampling. We have
$$
SE(\hat{\beta}_1)^2 = \frac{\sigma^2}{\sum_{i=1}^n (x_i - \bar{x})^2}, \quad SE(\hat{\beta}_0)^2 = \sigma^2 \left( \frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^n (x_i - \bar{x})^2} \right),
$$
where $\sigma^2 = \text{Var}(\epsilon)$.

These standard errors can be used to compute confidence intervals. A 95% confidence interval is defined as a range of values such that with 95% probability, the range will contain the true unknown value of the parameter. It has the form
$$
\hat{\beta}_1 \pm 2 \cdot SE(\hat{\beta}_1).
$$

Confidence interval $-$ continued
That is, there is approximately a 95% chance that the interval
$$
[\hat{\beta}_1 - 2 \cdot SE(\hat{\beta}_1), \hat{\beta}_1 + 2 \cdot SE(\hat{\beta}_1)]
$$
will contain the true value of $\beta_1$ (under a scenario where we got repeated samples like the present sample).

For the advertising data, the 95% confidence interval for $\beta_1$ is [0.042, 0.053].

### Hypothesis testing
Standard errors can also be used to perform **hypothesis tests** on the coefficients. The most common hypothesis test involves testing the **null hypothesis** of
$$
H_0: \text{There is no relationship between } X \text{ and } Y
$$
versus the **alternative hypothesis**
$$
H_A: \text{There is some relationship between } X \text{ and } Y.
$$

Mathematically, this corresponds to testing
$$
H_0: \beta_1 = 0
$$
versus
$$
H_A: \beta_1 \neq 0,
$$
since if $\beta_1 = 0$ then the model reduces to $Y = \beta_0 + \epsilon$, and $X$ is not associated with $Y$.

To test the null hypothesis, we compute a **t-statistic**, given by
$$
t = \frac{\hat{\beta}_1 - 0}{SE(\hat{\beta}_1)},
$$

This will have a t-distribution with $n - 2$ degrees of freedom, assuming $\beta_1 = 0$.

Using statistical software, it is easy to compute the probability of observing any value equal to $|t|$ or larger. We call this probability the **p-value**.

### Results for the advertising data

|           | Coefficient | Std. Error | t-statistic | p-value  |
|-----------|-------------|------------|-------------|----------|
| Intercept | 7.0325      | 0.4578     | 15.36       | < 0.0001 |
| TV        | 0.0475      | 0.0027     | 17.67       | < 0.0001 |

### Assessing the Overall Accuracy of the Model

We compute the **Residual Standard Error**
$$
RSE = \sqrt{\frac{1}{n - 2} RSS} = \sqrt{\frac{1}{n - 2} \sum_{i=1}^n (y_i - \hat{y}_i)^2},
$$
where the **residual sum-of-squares** is $RSS = \sum_{i=1}^n (y_i - \hat{y}_i)^2$.

**R-squared** or fraction of variance explained is
$$
R^2 = \frac{TSS - RSS}{TSS} = 1 - \frac{RSS}{TSS},
$$
where $TSS = \sum_{i=1}^n (y_i - \bar{y})^2$ is the total sum of squares.

It can be shown that in this simple linear regression setting that $R^2 = r^2$, where $r$ is the correlation between $X$ and $Y$:
$$
r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2 \sum_{i=1}^n (y_i - \bar{y})^2}}.
$$

### Advertising data results

| Quantity                | Value  |
|-------------------------|--------|
| Residual Standard Error | 3.26   |
| $R^2$                   | 0.612  |
| F-statistic             | 312.1  |

## Multiple Linear Regression

Here our model is

$$
Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \cdots + \beta_pX_p + \epsilon,
$$

We interpret $\beta_j$ as the average effect on $Y$ of a one unit increase in $X_j$, holding all other predictors fixed. In the advertising example, the model becomes

$$
\text{sales} = \beta_0 + \beta_1 \times \text{TV} + \beta_2 \times \text{radio} + \beta_3 \times \text{newspaper} + \epsilon.
$$

### Interpreting Regression Coefficients

The ideal scenario is when the predictors are uncorrelated — a **balanced design**:
- Each coefficient can be estimated and tested separately.
- Interpretations such as “a unit change in $X_j$ is associated with a $\beta_j$ change in $Y$, while all the other variables stay fixed”, are possible.

Correlations amongst predictors cause problems:
- The variance of all coefficients tends to increase, sometimes dramatically.
- Interpretations become hazardous — when $X_j$ changes, everything else changes.

**Claims of causality** should be avoided for observational data.

## The Woes of (Interpreting) Regression Coefficients

“Data Analysis and Regression” Mosteller and Tukey 1977:
- A regression coefficient $\beta_j$ estimates the expected change in $Y$ per unit change in $X_j$, with all other predictors held fixed. But predictors usually change together!

_Example:_ 
- $Y$: total amount of change in your pocket;
- $X_1$: number of coins;
- $X_2$: number of pennies, nickels, and dimes. By itself, the regression coefficient of $Y$ on $X_2$ will be > 0. But how about with $X_1$ in the model?

_Example:_
- $Y$: number of tackles by a football player in a season;
- $W$ and $H$: his weight and height.
- Fitted regression model is $\hat{Y} = \hat{\beta}_0 + 0.50W - 0.10H$. How do we interpret $\hat{\beta}_2 < 0$?

## Estimation and Prediction for Multiple Regression

Given estimates $\hat{\beta}_0, \hat{\beta}_1, \ldots, \hat{\beta}_p$, we can make predictions using the formula

$$
\hat{y} = \hat{\beta}_0 + \hat{\beta}_1x_1 + \hat{\beta}_2x_2 + \cdots + \hat{\beta}_px_p.
$$

We estimate $\beta_0, \beta_1, \ldots, \beta_p$ as the values that minimize the sum of squared residuals

$$
RSS = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n (y_i - \hat{\beta}_0 - \hat{\beta}_1x_{i1} - \hat{\beta}_2x_{i2} - \cdots - \hat{\beta}_px_{ip})^2.
$$

This is done using standard statistical software. The values $\hat{\beta}_0, \hat{\beta}_1, \ldots, \hat{\beta}_p$ that minimize $RSS$ are the multiple least squares regression coefficient estimates.

![img_3.png](figs/1/img_3.png)

## Results for Advertising Data

|           | Coefficient | Std. Error | t-statistic | p-value  |
|-----------|-------------|------------|-------------|----------|
| Intercept | 2.939       | 0.3119     | 9.42        | < 0.0001 |
| TV        | 0.046       | 0.0014     | 32.81       | < 0.0001 |
| radio     | 0.189       | 0.0086     | 21.89       | < 0.0001 |
| newspaper | -0.001      | 0.0059     | -0.18       | 0.8599   |

### Correlations

|           | TV     | radio  | newspaper | sales  |
|-----------|--------|--------|-----------|--------|
| TV        | 1.0000 | 0.0548 | 0.0567    | 0.7822 |
| radio     |        | 1.0000 | 0.3541    | 0.5762 |
| newspaper |        |        | 1.0000    | 0.2283 |
| sales     |        |        |           | 1.0000 |

## Important Questions

1. Is at least one of the predictors $X_1, X_2, \ldots, X_p$ useful in predicting the response?
2. Do all the predictors help to explain $Y$, or is only a subset of the predictors useful?
3. How well does the model fit the data?
4. Given a set of predictor values, what response value should we predict, and how accurate is our prediction?

## Is at Least One Predictor Useful?

For the first question, we can use the **F-statistic**

$$
F = \frac{(TSS - RSS)/p}{RSS/(n - p - 1)} \sim F_{p, n-p-1}
$$

| Quantity                | Value  |
|-------------------------|--------|
| Residual Standard Error | 1.69   |
| $R^2$                   | 0.897  |
| F-statistic             | 570    |

## Qualitative Predictors with More Than Two Levels
With more than two levels, we create additional dummy variables. For example, for the ethnicity variable we create two dummy variables. The first could be

$$
x_{i1} =
\begin{cases}
1 & \text{if } i\text{th person is Asian} \\
0 & \text{if } i\text{th person is not Asian}
\end{cases}
$$

and the second could be

$$
x_{i2} =
\begin{cases}
1 & \text{if } i\text{th person is Caucasian} \\
0 & \text{if } i\text{th person is not Caucasian}
\end{cases}
$$

Then both of these variables can be used in the regression equation, in order to obtain the model

$$
y_i = \beta_0 + \beta_1x_{i1} + \beta_2x_{i2} + \epsilon_i = 
\begin{cases} 
\beta_0 + \beta_1 + \epsilon_i & \text{if } i\text{th person is Asian} \\
\beta_0 + \beta_2 + \epsilon_i & \text{if } i\text{th person is Caucasian} \\
\beta_0 + \epsilon_i & \text{if } i\text{th person is AA}
\end{cases}
$$

There will always be one fewer dummy variable than the number of levels. The level with no dummy variable — African American in this example — is known as the baseline.

## Results for Ethnicity

|                      | Coefficient | Std. Error | t-statistic | p-value  |
|----------------------|-------------|------------|-------------|----------|
| Intercept            | 531.00      | 46.32      | 11.464      | < 0.0001 |
| ethnicity[Asian]     | -18.69      | 65.02      | -0.287      | 0.7740   |
| ethnicity[Caucasian] | -12.50      | 56.68      | -0.221      | 0.8260   |

## Extensions of the Linear Model

Removing the additive assumption: interactions and nonlinearity

### Interactions

In our previous analysis of the Advertising data, we assumed that the effect on sales of increasing one advertising medium is independent of the amount spent on the other media.

For example, the linear model

$$
\text{sales} = \beta_0 + \beta_1 \times \text{TV} + \beta_2 \times \text{radio} + \beta_3 \times \text{newspaper}
$$

states that the average effect on sales of a one-unit increase in TV is always $\beta_1$, regardless of the amount spent on radio.

Suppose that spending money on radio advertising actually increases the effectiveness of TV advertising, so that the slope term for TV should increase as radio increases.

In this situation, given a fixed budget of \$100,000, spending half on radio and half on TV may increase sales more than allocating the entire amount to either TV or to radio.

In marketing, this is known as a **synergy** effect, and in statistics, it is referred to as an **interaction** effect.

![img_4.png](figs/1/img_4.png)

### Modelling Interactions — Advertising Data

Model takes the form

$$
\text{sales} = \beta_0 + \beta_1 \times \text{TV} + \beta_2 \times \text{radio} + \beta_3 \times (\text{radio} \times \text{TV}) + \epsilon = \beta_0 + (\beta_1 + \beta_3 \times \text{radio}) \times \text{TV} + \beta_2 \times \text{radio} + \epsilon.
$$

Results:

|           | Coefficient | Std. Error | t-statistic | p-value  |
|-----------|-------------|------------|-------------|----------|
| Intercept | 6.7502      | 0.248      | 27.23       | < 0.0001 |
| TV        | 0.0191      | 0.002      | 12.70       | < 0.0001 |
| radio     | 0.0289      | 0.009      | 3.24        | 0.0014   |
| TV×radio  | 0.0011      | 0.000      | 20.73       | < 0.0001 |

The results in this table suggest that interactions are important. The p-value for the interaction term TV×radio is extremely low, indicating that there is strong evidence for $H_A : \beta_3 \neq 0$.

The $R^2$ for the interaction model is 96.8%, compared to only 89.7% for the model that predicts sales using TV and radio without an interaction term.

This means that $(96.8 - 89.7)/(100 - 89.7) = 69\%$ of the variability in sales that remains after fitting the additive model has been explained by the interaction term.

The coefficient estimates in the table suggest that an increase in TV advertising of \$1,000 is associated with increased sales of

$$
(\hat{\beta}_1 + \hat{\beta}_3 \times \text{radio}) \times 1000 = 19 + 1.1 \times \text{radio} \text{ units}.
$$

An increase in radio advertising of \$1,000 will be associated with an increase in sales of

$$
(\hat{\beta}_2 + \hat{\beta}_3 \times \text{TV}) \times 1000 = 29 + 1.1 \times \text{TV} \text{ units}.
$$

### Hierarchy

Sometimes it is the case that an interaction term has a very small p-value, but the associated main effects (in this case, TV and radio) do not.

The **hierarchy principle**:
If we include an interaction in a model, we should also include the main effects, even if the p-values associated with their coefficients are not significant.

The rationale for this principle is that interactions are hard to interpret in a model without main effects — their meaning is changed. Specifically, the interaction terms also contain main effects, if the model has no main effect terms.

### Interactions Between Qualitative and Quantitative Variables

Consider the Credit data set, and suppose that we wish to predict balance using income (quantitative) and student (qualitative).

**Without an interaction term**, the model takes the form

$$
\text{balance}_i \approx \beta_0 + \beta_1 \times \text{income}_i + 
\begin{cases} 
\beta_2 & \text{if } i\text{th person is a student} \\
0 & \text{if } i\text{th person is not a student}
\end{cases}
= \beta_1 \times \text{income}_i + 
\begin{cases} 
\beta_0 + \beta_2 & \text{if } i\text{th person is a student} \\
\beta_0 & \text{if } i\text{th person is not a student}
\end{cases}
$$

**With interactions**, it takes the form

$$
\text{balance}_i \approx \beta_0 + \beta_1 \times \text{income}_i + 
\begin{cases} 
\beta_2 + \beta_3 \times \text{income}_i & \text{if student} \\
0 & \text{if not student}
\end{cases}
= 
\begin{cases} 
(\beta_0 + \beta_2) + (\beta_1 + \beta_3) \times \text{income}_i & \text{if student} \\
\beta_0 + \beta_1 \times \text{income}_i & \text{if not student}
\end{cases}
$$

![img_5.png](figs/1/img_5.png)

### Non-linear Effects of Predictors

Polynomial regression on Auto data:
![img_6.png](figs/1/img_6.png)

The figure suggests that

$$
\text{mpg} = \beta_0 + \beta_1 \times \text{horsepower} + \beta_2 \times \text{horsepower}^2 + \epsilon
$$

may provide a better fit.

|              | Coefficient | Std. Error | t-statistic | p-value  |
|--------------|-------------|------------|-------------|----------|
| Intercept    | 56.9001     | 1.8004     | 31.6        | < 0.0001 |
| horsepower   | -0.4662     | 0.0311     | -15.0       | < 0.0001 |
| horsepower^2 | 0.0012      | 0.0001     | 10.1        | < 0.0001 |

# Classification

## Qualitative Variables

Qualitative variables take values in an unordered set $ C $, such as:
- $ \text{eye color} \in \{ \text{brown}, \text{blue}, \text{green} \} $
- $ \text{email} \in \{ \text{spam}, \text{ham} \} $

## Classification Task

Given a feature vector $ X $ and a qualitative response $ Y $ taking values in the set $ C $, the classification task is to build a function $ C(X) $ that takes as input the feature vector $ X $ and predicts its value for $ Y $; i.e. $ C(X) \in C $.

## Estimating Probabilities

Often we are more interested in estimating the probabilities that $ X $ belongs to each category in $ C $. For example, it is more valuable to have an estimate of the probability that an insurance claim is fraudulent, than a classification of fraudulent or not.

## Example: Credit Card Default

![img.png](figs/2/img.png)

### Default Classification Task

Suppose for the Default classification task that we code:

$$
Y =
\begin{cases} 
0 & \text{if No} \\
1 & \text{if Yes}
\end{cases}
$$

Can we simply perform a linear regression of $ Y $ on $ X $ and classify as Yes if $ \hat{Y} > 0.5 $?

- In this case of a binary outcome, linear regression does a good job as a classifier and is equivalent to linear discriminant analysis, which we discuss later.
- Since in the population $ E(Y | X = x) = \Pr(Y = 1 | X = x) $, we might think that regression is perfect for this task.
- However, linear regression might produce probabilities less than zero or greater than one. Logistic regression is more appropriate.

### Linear versus Logistic Regression

![img_1.png](figs/2/img_1.png)

The orange marks indicate the response $ Y $, either 0 or 1. Linear regression does not estimate $ \Pr(Y = 1 | X) $ well. Logistic regression seems well suited to the task.

Now suppose we have a response variable with three possible values. A patient presents at the emergency room, and we must classify them according to their symptoms.

$$
Y = 
\begin{cases} 
1 & \text{if stroke} \\
2 & \text{if drug overdose} \\
3 & \text{if epileptic seizure}
\end{cases}
$$

This coding suggests an ordering and in fact implies that the difference between stroke and drug overdose is the same as between drug overdose and epileptic seizure.

Linear regression is not appropriate here. Multiclass Logistic Regression or Discriminant Analysis are more appropriate.

## Logistic Regression

Let's write $p(X) = \Pr(Y = 1|X) $ for short and consider using balance to predict default. Logistic regression uses the form:

$$
p(X) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}} 
$$

where $e \approx 2.71828 $ is a mathematical constant (Euler's number). It is easy to see that no matter what values $\beta_0 $, $\beta_1 $, or $X $ take, $p(X) $ will have values between 0 and 1.

A bit of rearrangement gives:

$$ 
\log \left( \frac{p(X)}{1 - p(X)} \right) = \beta_0 + \beta_1 X 
$$

This monotone transformation is called the **log odds** or **logit** transformation of $p(X) $ (by log we mean natural log: ln).

### Linear versus Logistic Regression

![img_1.png](figs/2/img_1.png)

Logistic regression ensures that our estimate for $p(X) $ lies between 0 and 1.

## Maximum Likelihood

We use maximum likelihood to estimate the parameters.

$$
 L(\beta_0, \beta_1) = \prod_{i:y_i=1} p(x_i) \prod_{i:y_i=0} (1 - p(x_i)) 
$$

This **likelihood** gives the probability of the observed zeros and ones in the data. We pick $\beta_0$ and $\beta_1$ to maximize the likelihood of the observed data.

Most statistical packages can fit linear logistic regression models by maximum likelihood. In R, we use the `glm` function.

|           | Coefficient | Std. Error | Z-statistic | P-value  |
|-----------|-------------|------------|-------------|----------|
| Intercept | -10.6513    | 0.3612     | -29.5       | < 0.0001 |
| balance   | 0.0055      | 0.0002     | 24.9        | < 0.0001 |

## Making Predictions

What is our estimated probability of default for someone with a balance of \$1000?

$$
 \hat{p}(X) = \frac{e^{\hat{\beta}_0 + \hat{\beta}_1 X}}{1 + e^{\hat{\beta}_0 + \hat{\beta}_1 X}} = \frac{e^{-10.6513 + 0.0055 \times 1000}}{1 + e^{-10.6513 + 0.0055 \times 1000}} = 0.006 
$$

With a balance of \$2000?

$$
 \hat{p}(X) = \frac{e^{\hat{\beta}_0 + \hat{\beta}_1 X}}{1 + e^{\hat{\beta}_0 + \hat{\beta}_1 X}} = \frac{e^{-10.6513 + 0.0055 \times 2000}}{1 + e^{-10.6513 + 0.0055 \times 2000}} = 0.586 
$$

### Using Student as the Predictor

|              | Coefficient | Std. Error | Z-statistic | P-value  |
|--------------|-------------|------------|-------------|----------|
| Intercept    | -3.5041     | 0.0707     | -49.55      | < 0.0001 |
| student[Yes] | 0.4049      | 0.1150     | 3.52        | 0.0004   |

$$
 \widehat{Pr}(\text{default=Yes}|\text{student=Yes}) = \frac{e^{-3.5041 + 0.4049 \times 1}}{1 + e^{-3.5041 + 0.4049 \times 1}} = 0.0431 
$$

$$
 \widehat{Pr}(\text{default=Yes}|\text{student=No}) = \frac{e^{-3.5041 + 0.4049 \times 0}}{1 + e^{-3.5041 + 0.4049 \times 0}} = 0.0292 
$$

## Logistic Regression with Several Variables

$$
 \log \left( \frac{p(X)}{1 - p(X)} \right) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p 
$$

$$
 p(X) = \frac{e^{\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p}}{1 + e^{\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p}} 
$$

|              | Coefficient | Std. Error | Z-statistic | P-value  |
|--------------|-------------|------------|-------------|----------|
| Intercept    | -10.8690    | 0.4923     | -22.08      | < 0.0001 |
| balance      | 0.0057      | 0.0002     | 24.74       | < 0.0001 |
| income       | 0.0030      | 0.0082     | 0.37        | 0.7115   |
| student[Yes] | -0.6468     | 0.2362     | -2.74       | 0.0062   |

Why is the coefficient for student negative, while it was positive before?

## Confounding

![img_2.png](figs/2/img_2.png)

- Students tend to have higher balances than non-students, so their marginal default rate is higher than for non-students.
- But for each level of balance, students default less than non-students.
- Multiple logistic regression can tease this out.

## Example: South African Heart Disease

- 160 cases of MI (myocardial infarction) and 302 controls (all male in age range 15-64), from Western Cape, South Africa in early 80s.
- Overall prevalence very high in this region: 5.1%.
- Measurements on seven predictors (risk factors), shown in scatterplot matrix.
- Goal is to identify relative strengths and directions of risk factors.
- This was part of an intervention study aimed at educating the public on healthier diets.

![img_3.png](figs/2/img_3.png)

Scatterplot matrix of the South African Heart Disease data. The response is color coded — The cases (MI) are red, the controls turquoise. `famhist` is a binary variable, with 1 indicating family history of MI.

```r
> heartfit <- glm(chd ~ ., data = heart, family = binomial)
> summary(heartfit)
Call:
glm(formula = chd ~ ., family = binomial, data = heart)

Coefficients:
                 Estimate  Std. Error  z value  Pr(>|z|)
(Intercept)     -4.1295997  0.9641558  -4.283   1.84e-05 ***
sbp              0.0057607  0.0056326   1.023   0.30643
tobacco          0.0795256  0.0262150   3.034   0.00242 **
ldl              0.1847793  0.0574115   3.219   0.00129 **
famhistPresent   0.9391855  0.2248691   4.177   2.96e-05 ***
obesity         -0.0345434  0.0291053  -1.187   0.23529
alcohol          0.0006065  0.0044550   0.136   0.89171
age              0.0425412  0.0101749   4.181   2.90e-05 ***
(Dispersion parameter for binomial family taken to be 1)

Null deviance: 596.11 on 461 degrees of freedom
Residual deviance: 483.17 on 454 degrees of freedom
AIC: 499.17
```

## Case-control sampling and logistic regression

In South African data, there are 160 cases, 302 controls — $\tilde{\pi} = 0.35$ are cases. Yet the prevalence of MI in this region is $\pi = 0.05$.

With case-control samples, we can estimate the regression parameters $\beta_j$ accurately (if our model is correct); the constant term $\beta_0$ is incorrect.

We can correct the estimated intercept by a simple transformation: $$ \hat{\beta}_0^* = \hat{\beta}_0 + \log \left( \frac{\pi}{1 - \pi} \right) - \log \left( \frac{\tilde{\pi}}{1 - \tilde{\pi}} \right) $$

Often cases are rare and we take them all; up to five times that number of controls is sufficient.

## Diminishing returns in unbalanced binary data

Sampling more controls than cases reduces the variance of the parameter estimates. But after a ratio of about 5 to 1, the variance reduction flattens out.  

## Logistic regression with more than two classes

So far we have discussed logistic regression with two classes. It is easily generalized to more than two classes. One version (used in the R package glmnet) has the symmetric form: 

$$
\Pr(Y = k|X) = \frac{e^{\beta_{0k} + \beta_{1k}X_1 + \ldots + \beta_{pk}X_p}}{\sum_{j=1}^K e^{\beta_{0j} + \beta_{1j}X_1 + \ldots + \beta_{pj}X_p}} 
$$

Here there is a linear function for **each** class. 

(The mathier students will recognize that some cancellation is possible, and only $K - 1$ linear functions are needed as in 2-class logistic regression.) 

Multiclass logistic regression is also referred to as **multinomial regression**.