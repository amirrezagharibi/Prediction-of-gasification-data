# Gasification surrogate models: regression and ANN

Surrogate models that predict four outputs of a coal gasification process simulation from its four operating conditions, comparing regression baselines with multilayer perceptrons (MLPs).

|Inputs (operating conditions)|Outputs|
|-|-|
|Pressure (kPa), equivalence ratio (ER), moisture content (MC), temperature (°C)|H₂ (%), syngas LHV (MJ/kmol), efficiency η (%), λ (g/mol)|

## Method

* **Grouped split.** Rows with identical inputs stay in the same subset: about 56 % training, 14 % validation, 30 % test (10,000 rows).
* **No leakage.** Scalers are fitted on training rows only. All model choices, including a 36-configuration MLP search, are made on the validation rows before the test rows are used.
* **Final evaluation.** Selected settings are refitted on training + validation rows and scored once on the test rows.
* **Models.** Mean baseline, linear regression, two quadratic regressions, five hand-picked MLPs and the tuned MLP.

## Results

Mean normalised test RMSE across the four outputs (RMSE ÷ training standard deviation; lower is better):

|Model|Mean normalised RMSE|
|-|-|
|MLP tuned (selected on validation)|0.0097|
|Quadratic regression (selected)|0.0749|
|Linear regression|0.284|
|Mean baseline|0.988|

The best MLPs are close to each other on the test rows. Full per-output metrics are in the notebook.

## Important: how to read these results

The simulation varies all four operating conditions **together along one operating path**. Their pairwise correlations are 1.000, and one direction carries more than 99.9999 % of the input variance. All four are kept as inputs because they define each run, but this has consequences:

* The models learn the response **along this path**. They are not validated for other combinations of conditions.
* The separate effect of each input cannot be identified, so regression coefficients are not interpreted.
* The random test split measures **interpolation** between nearby points, which is why test R² is above 0.999.

The notebook therefore adds an **extrapolation check**: the selected models are refitted on the lower 85 % of the path and scored on the upper 15 %. The tuned MLP stays close (normalised RMSE 0.007–0.12 per output), while the quadratic regression drifts (0.11–0.54).

## 

