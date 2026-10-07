# Neural Network Portfolio Allocation

## Introduction and Motivation

This project explores the use of **neural networks for portfolio allocation** across multiple financial assets. The objective is to investigate whether a machine-learning-based allocation strategy can identify portfolio weights that improve risk-adjusted performance compared with a simple buy-and-hold benchmark.

Traditional portfolio strategies often require selecting asset weights based on historical estimates, optimization assumptions, or fixed allocation rules. A neural network provides an alternative approach by learning a relationship between recent market information and portfolio allocations. The model can therefore adapt its allocation as market conditions change rather than maintaining a fixed portfolio throughout the investment period.

However, maximizing portfolio returns alone can lead to impractical allocations. In particular, a model that frequently changes its portfolio weights may generate **high turnover**, resulting in increased transaction costs and unnecessary portfolio rebalancing. To address this, the strategy includes a **turnover penalty** that discourages large changes in portfolio weights between periods. This encourages the model to balance potential performance improvements against the practical cost of frequently changing the portfolio.

The strategy also restricts portfolio allocations to a specified **allocation range**. These constraints prevent the neural network from producing extreme portfolio weights and help ensure that the resulting portfolio remains practically implementable and appropriately diversified.

The overall goal is therefore not simply to maximize returns, but to develop a portfolio allocation strategy that considers **performance, risk, allocation constraints, and portfolio turnover simultaneously**.

---

## Strategy

The portfolio allocation strategy uses a neural network to determine the allocation weights assigned to each asset.

At each time step, the model receives historical market information and produces a set of portfolio weights:

```math
w_t = (w_{1,t}, w_{2,t}, \ldots, w_{N,t})
```

where $w_{i,t}$ represents the allocation to asset $i$ at time $t$.

The portfolio return is determined by the weighted returns of the individual assets:

```math
R_{p,t} = \sum_{i=1}^{N} w_{i,t}R_{i,t}
```

where $R_{i,t}$ is the return of asset $i$ at time $t$.

The model is trained using an objective that balances portfolio performance with the stability of the resulting allocations.

### Turnover Penalty

A key component of the objective is the turnover penalty. Turnover measures how much the portfolio allocation changes between consecutive periods:

```math
\text{Turnover}_t =
\sum_{i=1}^{N}
\left|w_{i,t} - w_{i,t-1}\right|
```

The optimization objective therefore incorporates a penalty proportional to turnover:

```math
\mathcal{L}
=
\text{Portfolio Objective}
+
\lambda \cdot \text{Turnover}
```

The parameter $\lambda$ controls the strength of this penalty.

A larger value of $\lambda$ makes the model more conservative about changing allocations, producing a more stable portfolio but potentially reducing responsiveness to changes in market conditions. A smaller value allows the model to change allocations more freely in pursuit of higher returns.

This provides a way to account for the practical consequences of frequent portfolio rebalancing rather than evaluating the strategy purely based on theoretical returns.

### Allocation Constraints

The model also imposes bounds on the portfolio weights. These constraints limit how much capital can be allocated to an individual asset and prevent the neural network from concentrating the portfolio excessively in a small number of assets.

The allocation constraints therefore provide a practical safeguard against extreme model outputs while still allowing the network to dynamically adjust the portfolio.

---

## Model and Hyperparameters

Several hyperparameters control how the neural network learns and how aggressively it adjusts the portfolio.

| Hyperparameter | Purpose |
|---|---|
| **Lookback window** | Controls how much historical return information is provided to the model when making allocation decisions. |
| **Hidden layers** | Determines the depth of the neural network and its ability to learn nonlinear relationships in the data. |
| **Hidden units** | Controls the number of neurons available within each hidden layer and therefore the model's representational capacity. |
| **Learning rate** | Controls the size of the updates made to the neural-network parameters during optimization. |
| **Batch size** | Determines how many observations are used for each parameter update during training. |
| **Number of epochs** | Controls how many times the model passes through the training data. |
| **Turnover penalty** | Controls how strongly the model is penalized for changing portfolio allocations between periods. |
| **Allocation bounds** | Restrict the minimum and maximum allocation permitted for each asset. |
| **Initial portfolio weights** | Determines the starting allocation used when calculating portfolio turnover. |

The hyperparameters allow the strategy to control the trade-off between **model flexibility, portfolio responsiveness, stability, and practical implementability**.

---

## Evaluation

The strategy is evaluated using both portfolio performance and risk measures.

Rather than evaluating the model solely on total return, the analysis considers:

- **Portfolio returns**
- **Value at Risk (VaR)**
- **Conditional Value at Risk (CVaR)**
- **Portfolio turnover**
- **Performance relative to a buy-and-hold benchmark**

The buy-and-hold portfolio provides a simple benchmark against which the neural-network strategy can be evaluated. Comparing both return and risk measures helps determine whether improvements in performance are accompanied by an undesirable increase in portfolio risk.

The evaluation is performed using separate training, validation, and testing periods where appropriate, with the testing period used to assess how the strategy performs on previously unseen data.

---

## Conclusion

This project demonstrates how neural networks can be used to create a **dynamic portfolio allocation strategy** while incorporating practical investment considerations.

The main advantage of the approach is that it allows portfolio allocations to adapt to changing market conditions while simultaneously controlling several sources of model risk. The allocation constraints prevent extreme portfolio concentrations, while the turnover penalty discourages excessive trading and makes the resulting strategy more realistic for practical implementation.

Evaluating the strategy using both returns and downside-risk measures also provides a more complete view of portfolio performance than return alone. Comparing the neural-network strategy with a buy-and-hold benchmark helps determine whether the additional complexity of dynamic machine-learning-based allocation provides meaningful benefits.

Overall, the project provides a framework for studying the intersection of **machine learning, portfolio optimization, and financial risk management**. It demonstrates how a predictive model can be incorporated into an investment strategy while explicitly accounting for constraints and costs that arise in real-world portfolio management.
