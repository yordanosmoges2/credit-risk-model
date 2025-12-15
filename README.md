## Credit Scoring Business Understanding

### Basel II and the need for interpretability and documentation
The Basel II Capital Accord emphasizes accurate measurement and management of credit risk, requiring financial institutions to justify how risk is assessed and how capital adequacy is maintained. As a result, credit risk models must be interpretable, transparent, and well-documented. In this project, interpretability is important so that model decisions can be explained to regulators, risk managers, and business stakeholders. Proper documentation ensures reproducibility, auditability, and compliance with regulatory expectations.

### Proxy target variable justification and business risks
The provided dataset does not include a direct loan default label because it originates from eCommerce transaction data rather than traditional credit repayment data. To enable supervised learning, a proxy target variable is required. In this project, customer engagement behavior derived from Recency, Frequency, and Monetary (RFM) metrics is used as a proxy for credit risk. Customers with low engagement are assumed to have a higher likelihood of default.
The main business risks of using a proxy target include misclassification of customers who are temporarily inactive but not truly risky, potential bias introduced by behavioral patterns, and the possibility of making incorrect lending decisions if the proxy does not fully represent true default behavior.

### Trade-offs between interpretable and complex models
Simple, interpretable models such as Logistic Regression with Weight of Evidence (WoE) are easier to explain, validate, and govern, making them suitable for regulated financial environments. However, they may fail to capture complex nonlinear relationships in the data. More complex models such as Gradient Boosting often provide higher predictive performance by modeling interactions and nonlinearities, but they are harder to interpret and require additional explainability tools and governance controls. In a regulated credit risk context, the trade-off involves balancing predictive accuracy with transparency, compliance, and operational trust.

