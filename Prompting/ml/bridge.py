#!/usr/bin/env python3
"""
scikit-learn model bridge
========================
A minimal helper that listens for newline-delimited JSON commands on stdin and
responds with newline-delimited JSON on stdout.

Supported commands
------------------
1. fit_model  – trains a new estimator and returns a `model_id` handle
   {"cmd":"fit_model", "klass":"RandomForestClassifier", "params":{...}, "X": [...], "y": [...]}

2. predict    – predicts using an existing estimator
   {"cmd":"predict", "model_id":"<uuid>", "X": [...]} → returns {predictions:[...]}

3. save_model – serialises an estimator via joblib
   {"cmd":"save_model", "model_id":"<uuid>", "path":"model.joblib", "compress":3}

4. load_model – deserialises an estimator and returns a fresh model_id
   {"cmd":"load_model", "path":"model.joblib"}

5. optimize_model – performs hyper-parameter optimisation with nested CV & bootstrap CI
   {"cmd":"optimize_model", "klass":"RandomForestClassifier", "X": [...], "y": [...]}

6. fit_stacking_model – trains a StackingClassifier and returns a `model_id` handle
   {"cmd":"fit_stacking_model", "base_estimators": [{"name": "estimator1", "klass": "RandomForestClassifier", "params": {...}}, {"name": "estimator2", "klass": "GradientBoostingClassifier", "params": {...}}], "final_estimator": {"klass": "RandomForestClassifier", "params": {...}}, "X": [...], "y": [...]}
"""
import sys
import json
import uuid
import traceback
import os

# External deps – scikit-learn & joblib are expected to be installed in the runtime env
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from joblib import dump, load
    from optuna.integration import OptunaSearchCV
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    from optuna.study import MaxTrialsCallback
    from optuna.trial import TrialState
    import mlflow
except ImportError as exc:
    sys.stderr.write("Required dependencies missing: {}\n".format(exc))
    sys.exit(1)

# In-memory registry of live models ➜ model_id → estimator instance
MODELS = {}

# third-party deps
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import bootstrap

# silence optuna logging to avoid polluting stdout
import optuna.logging as olog
olog.set_verbosity(olog.WARNING)


def _read_json_line():
    """Read one line from stdin and decode JSON; exits cleanly on EOF."""
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return json.loads(line)


def _write(payload):
    """Emit payload as compact JSON on stdout."""
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


# Factory for supported estimators ------------------------------------------------

def _new_estimator(klass, params):
    if klass == "RandomForestClassifier":
        estimator = RandomForestClassifier(**params)
    elif klass == "GradientBoostingClassifier":
        estimator = GradientBoostingClassifier(**params)
    elif klass == "LogisticRegression":
        estimator = LogisticRegression(**params)
    else:
        raise ValueError(f"Unsupported estimator class: {klass}")
    
    # Always wrap the estimator in a pipeline with a scaler
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', estimator)
    ])


# Command handlers ----------------------------------------------------------------

def cmd_fit_model(payload):
    # Start MLflow run for tracking
    with mlflow.start_run():
        klass = payload["klass"]
        params = payload.get("params", {})
        X = np.array(payload["X"], dtype=float)
        y = np.array(payload["y"], dtype=int)
        
        # Log parameters to MLflow
        mlflow.log_params({
            "klass": klass,
            "n_samples": len(X),
            "n_features": X.shape[1] if len(X.shape) > 1 else 1,
            **{f"param_{k}": v for k, v in params.items()}
        })
        
        est = _new_estimator(klass, params)
        est.fit(X, y)
        
        # Calculate and log basic metrics
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(est, X, y, cv=3, scoring="accuracy")
        mlflow.log_metric("cv_mean_accuracy", float(np.mean(cv_scores)))
        mlflow.log_metric("cv_std_accuracy", float(np.std(cv_scores)))
        
        # Log the model to the registry
        model_name = f"{klass.lower()}_simple_model"
        mlflow.sklearn.log_model(
            sk_model=est,
            artifact_path="model",
            registered_model_name=model_name,
            signature=mlflow.models.infer_signature(X, est.predict(X))
        )
        
        model_id = str(uuid.uuid4())
        MODELS[model_id] = est
        return {"model_id": model_id}


def cmd_predict(payload):
    model_id = payload["model_id"]
    if model_id not in MODELS:
        raise ValueError(f"Model ID {model_id} not found. Available models: {list(MODELS.keys())}")
    
    est = MODELS[model_id]
    X = payload["X"]
    
    # Input validation and type checking
    if not isinstance(X, list):
        raise TypeError(f"Expected X to be a list, got {type(X)}")
    
    if len(X) == 0:
        raise ValueError("Input X cannot be empty")
    
    # Convert to numpy array with validation
    try:
        X_array = np.array(X, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Could not convert X to numeric array: {e}")
    
    # Validate input shape
    if len(X_array.shape) == 1:
        X_array = X_array.reshape(1, -1)
    elif len(X_array.shape) != 2:
        raise ValueError(f"Expected 2D input array, got shape {X_array.shape}")
    
    # Check for invalid values
    if np.any(np.isnan(X_array)) or np.any(np.isinf(X_array)):
        raise ValueError("Input contains NaN or infinite values")
    
    # Make prediction with error handling
    try:
        if hasattr(est, "predict_proba"):
            preds = est.predict_proba(X_array).tolist()
            # Ensure we return both probabilities and raw predictions
            raw_preds = est.predict(X_array).tolist()
            return {
                "predictions": raw_preds,
                "probabilities": preds,
                "model_type": type(est).__name__
            }
        else:
            preds = est.predict(X_array).tolist()
            return {
                "predictions": preds,
                "model_type": type(est).__name__
            }
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")


def cmd_save_model(payload):
    model_id = payload["model_id"]
    path = payload["path"]
    compress = payload.get("compress", 3)
    dump(MODELS[model_id], path, compress=compress)
    return {"saved": True, "path": path}


def cmd_load_model(payload):
    path = payload["path"]
    est = load(path)
    model_id = str(uuid.uuid4())
    MODELS[model_id] = est
    return {"model_id": model_id}


def cmd_fit_stacking_model(payload):
    """Handles training for a StackingClassifier."""
    # Base estimators
    base_estimators_config = payload.get("base_estimators", [])
    base_estimators = [
        (_name, _new_estimator(_klass, _params))
        for _name, _klass, _params in base_estimators_config
    ]

    # Final estimator
    final_estimator_config = payload.get("final_estimator")
    final_estimator = _new_estimator(
        final_estimator_config["klass"],
        final_estimator_config.get("params", {})
    )

    # Stacking Classifier
    stacker = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=int(payload.get("cv", 5)),
        passthrough=payload.get("passthrough", False)
    )

    X = payload["X"]
    y = payload["y"]
    stacker.fit(X, y)

    model_id = str(uuid.uuid4())
    MODELS[model_id] = stacker
    return {"model_id": model_id}


HANDLERS = {
    "fit_model": cmd_fit_model,
    "predict": cmd_predict,
    "save_model": cmd_save_model,
    "load_model": cmd_load_model,
    "optimize_model": None,  # placeholder, will be assigned later
    "fit_stacking_model": cmd_fit_stacking_model,
}


# ---------------------------------------------------------------------
# Hyper-parameter optimisation with nested CV & bootstrap CI
# ---------------------------------------------------------------------

def _default_search_space(klass):
    """Return default Optuna distribution dict for supported estimator."""
    import optuna.distributions as od
    if klass == "RandomForestClassifier":
        return {
            "model__n_estimators": od.IntDistribution(50, 200, step=10, log=False),
            "model__max_depth": od.IntDistribution(3, 15, step=1, log=False),
            "model__min_samples_split": od.IntDistribution(2, 10, step=1, log=False),
        }
    if klass == "GradientBoostingClassifier":
        return {
            "model__n_estimators": od.IntDistribution(50, 300, step=25, log=False),
            "model__learning_rate": od.FloatDistribution(0.01, 0.3, log=True),
            "model__max_depth": od.IntDistribution(2, 6, step=1, log=False),
        }
    if klass == "LogisticRegression":
        return {
            "model__C": od.FloatDistribution(1e-3, 10.0, log=True),
            "model__penalty": od.CategoricalDistribution(["l2"]),
            "model__solver": od.CategoricalDistribution(["lbfgs"]),
            "model__max_iter": od.CategoricalDistribution([2000]), # Increase max_iter
        }
    raise ValueError(f"No default search space for {klass}")


def cmd_optimize_model(payload):
    # Start MLflow run for tracking
    with mlflow.start_run():
        klass = payload["klass"]
        X = np.array(payload["X"], dtype=float)
        y = np.array(payload["y"], dtype=int)

        # search space
        search_space = payload.get("search_space") or _default_search_space(klass)

        n_trials = int(payload.get("n_trials", 30))
        inner_folds = int(payload.get("inner_folds", 3))
        outer_folds = int(payload.get("outer_folds", 5))
        
        # Log main params to MLflow
        mlflow.log_params({
            "klass": klass,
            "n_trials": n_trials,
            "inner_folds": inner_folds,
            "outer_folds": outer_folds,
            "n_samples": len(X),
            "n_features": X.shape[1] if len(X.shape) > 1 else 1
        })

        # --- Persist study to SQLite database ---
        storage_dir = "./optuna_studies"
        os.makedirs(storage_dir, exist_ok=True)
        db_path = os.path.join(storage_dir, f"{klass}.log")
        storage = JournalStorage(JournalFileBackend(db_path))

        study = optuna.create_study(
            study_name=klass,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            load_if_exists=True
        )

        # OptunaSearchCV handles the inner CV and fitting
        est_for_search = _new_estimator(klass, {})
        cv_splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42)
        
        # Callback to retry failed trials for robustness
        retry_callback = MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL))

        optuna_search = OptunaSearchCV(
            estimator=est_for_search,
            param_distributions=search_space,
            n_trials=n_trials,
            cv=cv_splitter,
            scoring="accuracy",
            random_state=42,
            refit=True, # refit on the whole dataset to get best_estimator_
            verbose=0,
            study=study,
            callbacks=[retry_callback]
        )

        optuna_search.fit(X, y)
        
        # The best estimator is already fitted on the full training data
        best_estimator = optuna_search.best_estimator_
        best_params = optuna_search.best_params_

        # Store the fitted best estimator and get a handle for it
        model_id = str(uuid.uuid4())
        MODELS[model_id] = best_estimator

        # Nested CV outer loop to get unbiased performance estimate of the tuned model
        outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
        outer_scores = cross_val_score(best_estimator, X, y, cv=outer_cv, scoring="accuracy")

        # Bootstrap CI
        try:
            ci = bootstrap((outer_scores,), np.mean, confidence_level=0.95, n_resamples=1000)
            ci_low, ci_high = ci.confidence_interval
        except Exception:
            ci_low, ci_high = float(np.min(outer_scores)), float(np.max(outer_scores))

        # Replace NaN or inf with bounds
        if not np.isfinite(ci_low):
            ci_low = float(np.min(outer_scores))
        if not np.isfinite(ci_high):
            ci_high = float(np.max(outer_scores))

        # Log metrics and model to MLflow
        mlflow.log_metric("best_score", optuna_search.best_score_)
        mlflow.log_metric("mean_outer_score", float(np.mean(outer_scores)))
        mlflow.log_metric("ci_low", float(ci_low))
        mlflow.log_metric("ci_high", float(ci_high))
        mlflow.log_params(best_params)
        mlflow.log_artifact(db_path)
        
        # Log the model to the registry with semantic versioning
        model_name = f"{klass.lower()}_model"
        mlflow.sklearn.log_model(
            sk_model=best_estimator,
            artifact_path="model",
            registered_model_name=model_name,
            signature=mlflow.models.infer_signature(X, best_estimator.predict(X))
        )

        return {
            "model_id": model_id, # Return handle to the fitted model
            "best_params": best_params,
            "mean_outer_score": float(np.mean(outer_scores)),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }


# register handler
HANDLERS["optimize_model"] = cmd_optimize_model


def _define_search_space_for_stack(payload):
    """Dynamically construct a search space for a StackingClassifier."""
    import optuna.distributions as od
    space = {}
    
    # Search space for base estimators
    for est_cfg in payload.get("base_estimators", []):
        name = est_cfg["name"]
        klass = est_cfg["klass"]
        # Use default search space from single optimizer if not provided
        user_space = est_cfg.get("search_space") or _default_search_space(klass)
        for key, dist in user_space.items():
            # e.g., "rf__model__n_estimators"
            space[f"{name}__{key}"] = dist

    # Search space for final estimator
    final_cfg = payload.get("final_estimator", {})
    if final_cfg:
        name = "final_estimator"
        klass = final_cfg["klass"]
        user_space = final_cfg.get("search_space") or _default_search_space(klass)
        for key, dist in user_space.items():
            space[f"{name}__{key}"] = dist
            
    return space


def cmd_optimize_stacking_model(payload):
    """Hyper-parameter optimisation for a StackingClassifier."""
    with mlflow.start_run():
        X = np.array(payload["X"], dtype=float)
        y = np.array(payload["y"], dtype=int)

        # Log main params
        n_trials = int(payload.get("n_trials", 50))
        inner_folds = int(payload.get("inner_folds", 3))
        stacking_cv = int(payload.get("stacking_cv", 5))
        study_name = payload.get("study_name", "stacking_optimization")
        
        mlflow.log_params({
            "n_trials": n_trials,
            "inner_folds": inner_folds,
            "stacking_cv": stacking_cv,
            "study_name": study_name
        })

        # --- Build the StackingClassifier ---
        base_estimators_config = payload.get("base_estimators", [])
        base_estimators = [
            (cfg["name"], _new_estimator(cfg["klass"], {})) for cfg in base_estimators_config
        ]
        final_estimator_config = payload.get("final_estimator")
        final_estimator = _new_estimator(final_estimator_config["klass"], {})
        
        stacker = StackingClassifier(
            estimators=base_estimators,
            final_estimator=final_estimator,
            cv=stacking_cv
        )
        
        # --- Define search space ---
        search_space = _define_search_space_for_stack(payload)
        
        # --- Optuna setup ---
        storage_dir = "./optuna_studies"
        os.makedirs(storage_dir, exist_ok=True)
        db_path = os.path.join(storage_dir, f"{study_name}.log")
        storage = JournalStorage(JournalFileBackend(db_path))

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            load_if_exists=True
        )
        
        cv_splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42)
        retry_callback = MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL))

        optuna_search = OptunaSearchCV(
            estimator=stacker,
            param_distributions=search_space,
            n_trials=n_trials,
            cv=cv_splitter,
            scoring="accuracy",
            random_state=42,
            refit=True,
            verbose=0,
            study=study,
            callbacks=[retry_callback]
        )

        optuna_search.fit(X, y)
        
        best_estimator = optuna_search.best_estimator_
        model_id = str(uuid.uuid4())
        MODELS[model_id] = best_estimator

        # Log results
        mlflow.log_metric("best_score", optuna_search.best_score_)
        mlflow.log_params(optuna_search.best_params_)
        mlflow.log_artifact(db_path)
        
        # Log the model to the registry
        mlflow.sklearn.log_model(
            sk_model=best_estimator,
            artifact_path="stacking_model",
            registered_model_name=f"{study_name}_model"
        )

        return {
            "model_id": model_id,
            "best_params": optuna_search.best_params_,
            "best_score": optuna_search.best_score_,
        }


# register handler
HANDLERS["optimize_stacking_model"] = cmd_optimize_stacking_model


# Main loop -----------------------------------------------------------------------

while True:
    try:
        req = _read_json_line()
        cmd = req.get("cmd")
        if cmd not in HANDLERS:
            raise ValueError(f"Unknown command: {cmd}")
        result = HANDLERS[cmd](req)
        response = {"status": "ok", **result}
        # Echo request id back for the JS bridge to match promises
        if "__reqid" in req:
            response["__reqid"] = req["__reqid"]
        _write(response)
    except Exception as exc:
        err_resp = {"status": "error", "message": str(exc), "trace": traceback.format_exc()}
        if "__reqid" in req:
            err_resp["__reqid"] = req["__reqid"]
        _write(err_resp) 