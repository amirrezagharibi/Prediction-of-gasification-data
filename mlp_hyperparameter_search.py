"""Validation grid search for the gasification MLP surrogate."""

from itertools import product
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

HIDDEN_LAYER_OPTIONS = [(12,), (15,), (10, 10), (10, 15, 10)]
ALPHA_OPTIONS = [0.0000005, 0.0005, 0.005]
SOLVER_OPTIONS = [("lbfgs", None), ("adam", 0.001), ("adam", 0.005)]

# Adam stops when the training loss improves by less than `tol` for
# `n_iter_no_change` epochs. scikit-learn's default (tol=1e-4, 10 epochs) is
# coarse for standardised targets and stopped these runs after 18-55 epochs,
# without raising a ConvergenceWarning. A tighter tolerance lets them train.
ADAM_TOL = 1e-6
ADAM_N_ITER_NO_CHANGE = 20


def tune_mlp(train_x, train_y, validation_x, validation_y, seed=1):
    """Fit every grid configuration and rank it on the validation rows.

    Inputs and targets must be standardised with statistics from the training
    rows only. Because the targets are standardised, the RMSE of each target
    on this scale equals its RMSE divided by the training standard deviation
    (ddof=0), so the score matches the notebook's normalised RMSE.

    Returns the ranked trial table, the fitted best model and the warnings
    recorded while fitting it.
    """
    trials = []
    best_score = np.inf
    best_model = None
    best_warnings = []

    with threadpool_limits(limits=1):
        for layers, alpha, (solver, learning_rate) in product(
            HIDDEN_LAYER_OPTIONS, ALPHA_OPTIONS, SOLVER_OPTIONS
        ):
            params = {
                "hidden_layer_sizes": layers,
                "alpha": alpha,
                "solver": solver,
                "activation": "relu",
                "max_iter": 2000,
                "random_state": seed,
                "early_stopping": False,
            }
            if solver == "adam":
                params.update(
                    learning_rate_init=learning_rate,
                    tol=ADAM_TOL,
                    n_iter_no_change=ADAM_N_ITER_NO_CHANGE,
                )

            model = MLPRegressor(**params)
            with warnings.catch_warnings(record=True) as fit_warnings:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(train_x, train_y)

            errors = validation_y - model.predict(validation_x)
            rmse_by_target = np.sqrt(np.mean(errors ** 2, axis=0))
            score = float(np.mean(rmse_by_target))
            trials.append({
                "Hidden_layers": layers,
                "Solver": solver,
                "Alpha": alpha,
                "Initial_learning_rate": learning_rate,
                "Iterations": model.n_iter_,
                "Convergence_warning": any(
                    issubclass(w.category, ConvergenceWarning) for w in fit_warnings
                ),
                "Mean_normalised_RMSE": score,
            })

            if np.isfinite(score) and score < best_score:
                best_score = score
                best_model = model
                best_warnings = fit_warnings

    if best_model is None:
        raise ValueError("The MLP search produced no finite validation score.")

    search_table = pd.DataFrame(trials).sort_values(
        "Mean_normalised_RMSE", kind="stable", ignore_index=True
    )
    return search_table, best_model, best_warnings
