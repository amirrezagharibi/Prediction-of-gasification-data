# Gasification modelling: regression, ANN and TOPSIS

Two analyses of a gasification process simulation with four operating conditions and four performance outputs:

1. **Surrogate models** that predict the outputs from the operating conditions, comparing regression baselines with multilayer perceptrons (MLPs).
2. **TOPSIS ranking** of the operating conditions by their simulated performance.

|Inputs (operating conditions)|Outputs (performance)|
|-|-|
|Pressure (kPa), equivalence ratio (ER), moisture content (MC), temperature (°C)|H₂ concentration (%), syngas LHV (MJ/kmol), efficiency η (%), normalised CO₂ emission λ (g/mol)|

## Surrogate models

### Method

* **Grouped split.** Rows with identical inputs stay in the same subset: about 56 % training, 14 % validation, 30 % test (10,000 rows).
* **No leakage.** Scalers are fitted on training rows only. All model choices, including a 36-configuration MLP search, are made on the validation rows before the test rows are used.
* **Final evaluation.** Selected settings are refitted on training + validation rows and scored once on the test rows.
* **Models.** Mean baseline, linear regression, two quadratic regressions, five hand-picked MLPs and the tuned MLP.
* **Outputs.** Regression models are fitted separately for each output. Each MLP is a single network that predicts all four outputs, trained on standardised targets.

### Results

Mean normalised test RMSE across the four outputs (RMSE ÷ training standard deviation; lower is better):

|Model|Mean normalised RMSE|
|-|-|
|MLP tuned (selected on validation)|0.0097|
|Quadratic regression (selected)|0.0749|
|Linear regression|0.284|
|Mean baseline|0.988|

The best MLPs are close to each other on the test rows. Full per-output metrics are in the notebook.

### Important: how to read these results

The simulation varies all four operating conditions **together along one operating path**. Their pairwise correlations are 1.000, and one direction carries more than 99.9999 % of the input variance. All four are kept as inputs because they define each run, but this has consequences:

* The models learn the response **along this path**. They are not validated for other combinations of conditions.
* The separate effect of each input cannot be identified, so regression coefficients are not interpreted.
* The random test split measures **interpolation** between nearby points, which is why test R² is above 0.999.

The notebook therefore adds an **extrapolation check**: the selected models are refitted on the lower 85 % of the path and scored on the upper 15 %. The tuned MLP stays close (normalised RMSE 0.007–0.12 per output), while the quadratic regression drifts (0.11–0.54).

## TOPSIS ranking

Each unique operating condition is an alternative (9,221 after averaging repeated runs), scored on the four outputs: H₂, LHV and η are better when higher, CO₂ emission λ when lower. The operating conditions are not used as criteria: because they all rise together, ranking on them would only reproduce the order of the operating path.

* **Weights.** Equal, standard-deviation and CRITIC weights, all computed from min–max normalised criteria so they do not depend on units. CRITIC also discounts criteria that carry overlapping information (here LHV, η and λ).
* **Checks.** PyMCDM scores are verified against a NumPy implementation, and the best alternative is confirmed to be non-dominated.
* **Sensitivity.** The best condition is compared across weighting schemes and traced as the weight on H₂ changes.

**Findings.** The best conditions lie in the early-to-middle part of the operating path (around 260–390 kPa depending on the weights), where efficiency and LHV are high and CO₂ emission is low. The end of the path ranks worst: H₂ is highest there, but LHV and efficiency are lowest and CO₂ emission is highest. Giving H₂ more weight moves the best condition further along the path.

