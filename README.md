## Credit Scoring Business Understanding
### Basel II and why interpretability + documentation matter
Basel II emphasizes measuring and managing credit risk using evidence-based approaches and requires banks to hold capital relative to the risk they take. This pushes us to build models that are explainable, auditable, and well-documented. In practice, that means:
- We must clearly define the target (“default” or a proxy), the assumptions behind it, and how features are created.
- We should prefer models whose behavior can be explained to risk, compliance, and business stakeholders.
- We must keep full traceability (data version, code version, model version, metrics) so results can be validated and reproduced.

### Why a proxy target variable is necessary and its risks
Our dataset does not contain a direct label like “loan default” because it is transaction data from an eCommerce platform, not loan repayment history. To train any supervised model, we need a target variable, so we create a proxy label based on observable behavior (e.g., RFM: Recency, Frequency, Monetary value).
Business risks of using a proxy:
- Proxy ≠ true default: We may misclassify customers who are inactive for non-risk reasons (seasonality, one-time buyers).
- Bias risk: Behavioral proxies may disadvantage certain groups if spending patterns reflect access or lifestyle rather than willingness/ability to repay.
- Operational risk: If the proxy is weak, approvals/limits may be wrong, causing higher losses or rejecting good customers.

### Trade-offs: simple interpretable model vs complex high-performance model
In regulated financial use cases, we balance predictive performance with interpretability and governance.
- Simple models (e.g., Logistic Regression + WoE):
  - Pros: more explainable, easier to document, easier to monitor, typically preferred by regulators.
  - Cons: may underperform if relationships are nonlinear or interactions are important.
- Complex models (e.g., Gradient Boosting):
  - Pros: often higher accuracy/ROC-AUC by capturing nonlinearity and interactions.
  - Cons: harder to explain; may require extra governance tools (feature importance, SHAP), stronger monitoring, and more careful documentation to justify decisions.
A pragmatic approach is to train both and select the best model that meets performance goals while remaining acceptable for risk governance.
