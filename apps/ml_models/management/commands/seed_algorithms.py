from django.core.management.base import BaseCommand
from django.db import transaction

from apps.ml_models.models import Algorithm, AlgorithmFamily, AlgorithmTutorialSection


def _json_schema_base(title: str, description: str) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "additionalProperties": False,
        "description": description,
        "properties": {},
        "required": [],
    }


def _add_prop(schema: dict, name: str, prop: dict, required: bool = False) -> None:
    schema["properties"][name] = prop
    if required:
        schema["required"].append(name)


@transaction.atomic
def seed_algorithms() -> dict:
    """
    Idempotent seeding: update_or_create by key.
    Returns {created: int, updated: int}.
    """
    created = 0
    updated = 0

    algorithms = []

    # 1) Linear Regression (Simple)
    lr_schema = _json_schema_base(
        "LinearRegression Hyperparameters",
        "Hyperparameters for sklearn.linear_model.LinearRegression. "
        "Note: simple vs multiple linear regression differs by number of input features, not by estimator.",
    )
    _add_prop(lr_schema, "fit_intercept", {"type": "boolean", "default": True, "description": "Whether to calculate the intercept for this model."})
    _add_prop(lr_schema, "copy_X", {"type": "boolean", "default": True, "description": "If True, X will be copied; else it may be overwritten."})
    _add_prop(lr_schema, "n_jobs", {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}], "default": None, "description": "Number of jobs for computation. None means 1 unless in a joblib.parallel_backend context."})
    _add_prop(lr_schema, "positive", {"type": "boolean", "default": False, "description": "When True, forces coefficients to be positive (useful for interpretability constraints)."})

    algorithms.append({
        "key": "linear-regression-simple",
        "display_name": "Linear Regression (Simple)",
        "family": AlgorithmFamily.REGRESSION,
        "sklearn_class_path": "sklearn.linear_model.LinearRegression",
        "supported_tasks": ["REGRESSION"],
        "default_hyperparameters": {"fit_intercept": True, "copy_X": True, "n_jobs": None, "positive": False},
        "hyperparameter_schema": lr_schema,
        "educational_summary": (
            "Simple Linear Regression models a linear relationship between one input feature X and a continuous target y "
            "by fitting a straight line that minimizes squared prediction errors."
        ),
        "educational_details_md": (
            "### Intuition\n"
            "Simple Linear Regression assumes the target is approximately a **straight-line function** of one feature:\n"
            "\n"
            "$$\\hat{y} = \\beta_0 + \\beta_1 x$$\n"
            "\n"
            "The parameters $(\\beta_0, \\beta_1)$ are chosen to minimize the **sum of squared residuals**:\n"
            "\n"
            "$$\\min_{\\beta_0,\\beta_1} \\sum_{i=1}^{n}(y_i - (\\beta_0 + \\beta_1 x_i))^2$$\n"
            "\n"
            "### What the model learns\n"
            "- **Intercept** $(\\beta_0)$: predicted value when $x=0$.\n"
            "- **Slope** $(\\beta_1)$: change in predicted $y$ for a one-unit increase in $x$.\n"
            "\n"
            "### When it works well\n"
            "- Relationship is roughly linear.\n"
            "- Errors have roughly constant variance.\n"
            "- Outliers are limited (squared loss is sensitive to them).\n"
        ),
        "prerequisites_md": "Basic algebra, concept of a function, and idea of minimizing an error.",
        "typical_use_cases_md": "Trend estimation, baseline regression model, interpretability-focused regression.",
        "strengths_md": "Fast, interpretable coefficients, good baseline.",
        "limitations_md": "Cannot capture non-linear patterns unless features are engineered; sensitive to outliers.",
        "references": [
            {"title": "An Introduction to Statistical Learning", "author": "James, Witten, Hastie, Tibshirani", "year": 2021},
            {"title": "scikit-learn: LinearRegression documentation", "author": "scikit-learn developers", "year": 2025},
        ],
        "sections": [
            ("Model Form", 0, "We assume $$\\hat{y}=\\beta_0+\\beta_1x$$ and learn parameters that minimize squared error."),
            ("Loss Function", 1, "Least squares penalizes large errors heavily because errors are squared."),
            ("Interpreting Coefficients", 2, "Slope is expected change in y when x increases by 1 unit."),
        ]
    })

    # 1) Linear Regression (Multiple)
    algorithms.append({
        "key": "linear-regression-multiple",
        "display_name": "Linear Regression (Multiple)",
        "family": AlgorithmFamily.REGRESSION,
        "sklearn_class_path": "sklearn.linear_model.LinearRegression",
        "supported_tasks": ["REGRESSION"],
        "default_hyperparameters": {"fit_intercept": True, "copy_X": True, "n_jobs": None, "positive": False},
        "hyperparameter_schema": lr_schema,
        "educational_summary": (
            "Multiple Linear Regression extends linear regression to multiple input features, learning a linear combination "
            "of features to predict a continuous target."
        ),
        "educational_details_md": (
            "### Model\n"
            "With features $x_1, x_2, ..., x_p$:\n"
            "\n"
            "$$\\hat{y} = \\beta_0 + \\beta_1 x_1 + \\cdots + \\beta_p x_p$$\n"
            "\n"
            "The training objective remains least squares. The difference from simple regression is **vector-valued input**.\n"
            "\n"
            "### Practical notes\n"
            "- Feature scaling is not required for ordinary least squares, but can help numerics.\n"
            "- Strongly correlated features (multicollinearity) can make coefficients unstable.\n"
        ),
        "prerequisites_md": "Vectors, dot product, basic matrix intuition (helpful but not mandatory).",
        "typical_use_cases_md": "Multi-factor prediction, interpretable regression with multiple variables.",
        "strengths_md": "Interpretable, fast, good baseline.",
        "limitations_md": "Linear decision surface; sensitive to multicollinearity and outliers.",
        "references": [
            {"title": "The Elements of Statistical Learning", "author": "Hastie, Tibshirani, Friedman", "year": 2009},
            {"title": "scikit-learn: LinearRegression documentation", "author": "scikit-learn developers", "year": 2025},
        ],
        "sections": [
            ("Vector Form", 0, "Write $$\\hat{y}=\\beta_0+\\mathbf{x}^T\\boldsymbol{\\beta}$$ for compact representation."),
            ("Multicollinearity", 1, "Highly correlated features can inflate variance of coefficient estimates."),
        ]
    })

    # 2) Logistic Regression
    logreg_schema = _json_schema_base(
        "LogisticRegression Hyperparameters",
        "Hyperparameters for sklearn.linear_model.LogisticRegression (classification). "
        "Some solver/penalty combinations are constrained; enforce in service-layer validation.",
    )
    _add_prop(logreg_schema, "C", {"type": "number", "exclusiveMinimum": 0.0, "default": 1.0, "description": "Inverse regularization strength (smaller => stronger regularization)."})
    _add_prop(logreg_schema, "penalty", {"type": "string", "enum": ["l2", "l1", "elasticnet", "none"], "default": "l2", "description": "Norm used in the penalization."})
    _add_prop(logreg_schema, "solver", {"type": "string", "enum": ["lbfgs", "liblinear", "sag", "saga", "newton-cg"], "default": "lbfgs", "description": "Optimization algorithm."})
    _add_prop(logreg_schema, "max_iter", {"type": "integer", "minimum": 50, "default": 100, "description": "Maximum number of iterations for solver convergence."})
    _add_prop(logreg_schema, "fit_intercept", {"type": "boolean", "default": True})
    _add_prop(logreg_schema, "class_weight", {"anyOf": [{"type": "string", "enum": ["balanced"]}, {"type": "null"}], "default": None, "description": "Set to 'balanced' to adjust weights inversely proportional to class frequencies."})
    _add_prop(logreg_schema, "l1_ratio", {"anyOf": [{"type": "number", "minimum": 0.0, "maximum": 1.0}, {"type": "null"}], "default": None, "description": "Only used if penalty='elasticnet'."})

    algorithms.append({
        "key": "logistic-regression",
        "display_name": "Logistic Regression",
        "family": AlgorithmFamily.CLASSIFICATION,
        "sklearn_class_path": "sklearn.linear_model.LogisticRegression",
        "supported_tasks": ["CLASSIFICATION"],
        "default_hyperparameters": {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 100, "fit_intercept": True, "class_weight": None, "l1_ratio": None},
        "hyperparameter_schema": logreg_schema,
        "educational_summary": (
            "Logistic Regression is a linear classifier that models the probability of a class using the sigmoid function "
            "and learns weights by optimizing a log-loss objective with regularization."
        ),
        "educational_details_md": (
            "### Probability model\n"
            "For binary classification, Logistic Regression models:\n"
            "\n"
            "$$P(y=1\\mid \\mathbf{x})=\\sigma(\\beta_0 + \\mathbf{x}^T\\boldsymbol{\\beta})$$\n"
            "\n"
            "where $\\sigma(z)=\\frac{1}{1+e^{-z}}$ is the **sigmoid**.\n"
            "\n"
            "### Training objective\n"
            "We minimize **log loss** (cross-entropy), often with regularization:\n"
            "- **L2** encourages small weights (smooth decision boundary)\n"
            "- **L1** encourages sparsity (feature selection behavior)\n"
            "\n"
            "### Decision boundary\n"
            "Because it is linear in $\\mathbf{x}$, the boundary is a **hyperplane**.\n"
        ),
        "prerequisites_md": "Basic probability (odds), logs, and linear algebra basics.",
        "typical_use_cases_md": "Spam detection, credit risk, medical diagnosis (interpretable classification).",
        "strengths_md": "Interpretable, strong baseline, outputs probabilities.",
        "limitations_md": "Linear boundary unless features are engineered; can underfit complex patterns.",
        "references": [
            {"title": "Pattern Recognition and Machine Learning", "author": "Christopher Bishop", "year": 2006},
            {"title": "scikit-learn: LogisticRegression documentation", "author": "scikit-learn developers", "year": 2025},
        ],
        "sections": [
            ("Sigmoid & Odds", 0, "Logistic Regression turns linear scores into probabilities via the sigmoid."),
            ("Regularization", 1, "Regularization controls overfitting by penalizing large weights."),
            ("Interpreting Coefficients", 2, "A coefficient reflects how a feature shifts log-odds of the positive class."),
        ]
    })

    # 3) Decision Trees (Classifier & Regressor)
    dt_schema = _json_schema_base(
        "DecisionTree Hyperparameters",
        "Hyperparameters shared across sklearn.tree.DecisionTreeClassifier/Regressor.",
    )
    _add_prop(dt_schema, "criterion", {"type": "string", "enum": ["gini", "entropy", "log_loss", "squared_error", "friedman_mse", "absolute_error", "poisson"], "default": "gini",
                                      "description": "Split quality measure. Use classification criteria for classifier; regression criteria for regressor."})
    _add_prop(dt_schema, "max_depth", {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}], "default": None, "description": "Maximum tree depth; None grows until pure or min_samples constraints."})
    _add_prop(dt_schema, "min_samples_split", {"type": "integer", "minimum": 2, "default": 2})
    _add_prop(dt_schema, "min_samples_leaf", {"type": "integer", "minimum": 1, "default": 1})
    _add_prop(dt_schema, "max_features", {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0}, {"type": "string", "enum": ["sqrt", "log2"]}, {"type": "null"}],
                                         "default": None,
                                         "description": "Number of features to consider at each split."})
    _add_prop(dt_schema, "random_state", {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": 42})
    _add_prop(dt_schema, "splitter", {"type": "string", "enum": ["best", "random"], "default": "best"})

    algorithms.append({
        "key": "decision-tree-classifier",
        "display_name": "Decision Tree (Classifier)",
        "family": AlgorithmFamily.CLASSIFICATION,
        "sklearn_class_path": "sklearn.tree.DecisionTreeClassifier",
        "supported_tasks": ["CLASSIFICATION"],
        "default_hyperparameters": {"criterion": "gini", "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": None, "random_state": 42, "splitter": "best"},
        "hyperparameter_schema": dt_schema,
        "educational_summary": "A Decision Tree classifier splits data using rules (feature thresholds) to minimize impurity and form a tree of decisions.",
        "educational_details_md": (
            "### How it works\n"
            "A Decision Tree repeatedly chooses a feature and a split point that best separates classes.\n"
            "\n"
            "- **Nodes**: tests like `feature_j <= threshold`\n"
            "- **Leaves**: class prediction (often majority class)\n"
            "\n"
            "### Impurity (classification)\n"
            "- **Gini**: measures how mixed the classes are in a node\n"
            "- **Entropy**: information-theoretic measure of uncertainty\n"
            "\n"
            "### Overfitting control\n"
            "Trees can overfit by growing too deep. Use **max_depth**, **min_samples_leaf**, etc.\n"
        ),
        "prerequisites_md": "Understanding of if/else rules and basic classification.",
        "typical_use_cases_md": "Interpretable classification, feature importance exploration.",
        "strengths_md": "Human-readable rules, handles non-linear splits, little preprocessing needed.",
        "limitations_md": "High variance (unstable), overfits without pruning/constraints.",
        "references": [
            {"title": "Classification and Regression Trees", "author": "Breiman et al.", "year": 1984},
            {"title": "scikit-learn: DecisionTreeClassifier documentation", "author": "scikit-learn developers", "year": 2025},
        ],
        "sections": [
            ("Splitting", 0, "Choose splits that reduce impurity the most."),
            ("Stopping Rules", 1, "Limit depth or require minimum samples to split to reduce overfitting."),
        ]
    })

    algorithms.append({
        "key": "decision-tree-regressor",
        "display_name": "Decision Tree (Regressor)",
        "family": AlgorithmFamily.REGRESSION,
        "sklearn_class_path": "sklearn.tree.DecisionTreeRegressor",
        "supported_tasks": ["REGRESSION"],
        "default_hyperparameters": {"criterion": "squared_error", "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": None, "random_state": 42, "splitter": "best"},
        "hyperparameter_schema": dt_schema,
        "educational_summary": "A Decision Tree regressor splits the feature space into regions and predicts the average target value in each region.",
        "educational_details_md": (
            "### Regression view\n"
            "The tree partitions the space into rectangles (regions). For any region $R$, prediction is usually:\n"
            "\n"
            "$$\\hat{y}(x)=\\frac{1}{|R|}\\sum_{i\\in R} y_i$$\n"
            "\n"
            "Splits minimize an error criterion such as **squared error**.\n"
        ),
        "prerequisites_md": "Basic regression concept and averages.",
        "typical_use_cases_md": "Non-linear regression with interpretability, feature interaction exploration.",
        "strengths_md": "Captures non-linearities, minimal preprocessing.",
        "limitations_md": "Piecewise-constant predictions; can be unstable; can overfit.",
        "references": [
            {"title": "The Elements of Statistical Learning", "author": "Hastie, Tibshirani, Friedman", "year": 2009},
            {"title": "scikit-learn: DecisionTreeRegressor documentation", "author": "scikit-learn developers", "year": 2025},
        ],
        "sections": [
            ("Regions", 0, "Each leaf corresponds to a region with a constant predicted value."),
            ("Bias-Variance", 1, "Deeper trees reduce bias but increase variance."),
        ]
    })

    # 4) Naive Bayes (Gaussian/Multinomial/Bernoulli)
    gnb_schema = _json_schema_base("GaussianNB Hyperparameters", "Hyperparameters for sklearn.naive_bayes.GaussianNB.")
    _add_prop(gnb_schema, "var_smoothing", {"type": "number", "minimum": 0.0, "default": 1e-9, "description": "Portion of largest variance added for stability."})

    algorithms.append({
        "key": "naive-bayes-gaussian",
        "display_name": "Naive Bayes (Gaussian)",
        "family": AlgorithmFamily.CLASSIFICATION,
        "sklearn_class_path": "sklearn.naive_bayes.GaussianNB",
        "supported_tasks": ["CLASSIFICATION"],
        "default_hyperparameters": {"var_smoothing": 1e-9},
        "hyperparameter_schema": gnb_schema,
        "educational_summary": "Gaussian Naive Bayes models each feature with a normal distribution per class and assumes features are conditionally independent given the class.",
        "educational_details_md": (
            "### Bayes rule\n"
            "We predict the class $c$ that maximizes:\n"
            "\n"
            "$$P(c\\mid \\mathbf{x}) \\propto P(c)\\prod_j P(x_j\\mid c)$$\n"
            "\n"
            "### Naive (independence) assumption\n"
            "Features are assumed independent given the class. This is often false, but surprisingly effective.\n"
            "\n"
            "### Gaussian likelihood\n"
            "For continuous features, each $x_j\\mid c$ is modeled as a normal distribution.\n"
        ),
        "prerequisites_md": "Bayes rule basics, probability distributions (normal distribution).",
        "typical_use_cases_md": "Fast baseline classifier, text/sensor data after appropriate feature choice.",
        "strengths_md": "Very fast training/prediction; works well with small data.",
        "limitations_md": "Independence assumption can limit accuracy; probability calibration may be poor.",
        "references": [
            {"title": "Pattern Recognition and Machine Learning", "author": "Christopher Bishop", "year": 2006},
            {"title": "scikit-learn: GaussianNB documentation", "author": "scikit-learn developers", "year": 2025},
        ],
        "sections": [
            ("Independence Assumption", 0, "We multiply per-feature likelihoods, assuming conditional independence."),
            ("Why it can work", 1, "Even if independence is false, decision boundaries can still be useful."),
        ]
    })

    mnb_schema = _json_schema_base("MultinomialNB Hyperparameters", "Hyperparameters for sklearn.naive_bayes.MultinomialNB (counts/frequencies).")
    _add_prop(mnb_schema, "alpha", {"type": "number", "minimum": 0.0, "default": 1.0, "description": "Additive smoothing (Laplace smoothing)."})
    _add_prop(mnb_schema, "fit_prior", {"type": "boolean", "default": True})
    _add_prop(mnb_schema, "force_alpha", {"type": "boolean", "default": True, "description": "If True, alpha is not automatically adjusted."})

    algorithms.append({
        "key": "naive-bayes-multinomial",
        "display_name": "Naive Bayes (Multinomial)",
        "family": AlgorithmFamily.CLASSIFICATION,
        "sklearn_class_path": "sklearn.naive_bayes.MultinomialNB",
        "supported_tasks": ["CLASSIFICATION"],
        "default_hyperparameters": {"alpha": 1.0, "fit_prior": True, "force_alpha": True},
        "hyperparameter_schema": mnb_schema,
        "educational_summary": "Multinomial Naive Bayes is designed for discrete counts (e.g., word counts) and is common in text classification.",
        "educational_details_md": (
            "### Count-based likelihood\n"
            "MultinomialNB models features as counts (or TF/TF-IDF non-negative values). "
            "It estimates class-conditional feature probabilities and applies Bayes rule.\n"
            "\n"
            "### Smoothing\n"
            "**Alpha** prevents zero probabilities for unseen features, improving generalization.\n"
        ),
        "prerequisites_md": "Bayes rule and understanding of discrete counts/frequencies.",
        "typical_use_cases_md": "Spam filtering, topic classification, sentiment analysis with bag-of-words features.",
        "strengths_md": "Very strong baseline for text; extremely fast.",
        "limitations_md": "Requires non-negative features; independence assumption remains.",
        "references": [
            {"title": "Speech and Language Processing", "author": "Jurafsky & Martin", "year": 2023},
            {"title": "scikit-learn: MultinomialNB documentation", "author": "scikit-learn developers", "year": 2025},
        ],
        "sections": [
            ("Smoothing", 0, "Additive smoothing avoids zero probabilities and improves robustness."),
        ]
    })

    bnb_schema = _json_schema_base("BernoulliNB Hyperparameters", "Hyperparameters for sklearn.naive_bayes.BernoulliNB (binary features).")
    _add_prop(bnb_schema, "alpha", {"type": "number", "minimum": 0.0, "default": 1.0})
    _add_prop(bnb_schema, "fit_prior", {"type": "boolean", "default": True})
    _add_prop(bnb_schema, "binarize", {"anyOf": [{"type": "number"}, {"type": "null"}], "default": 0.0, "description": "Threshold for binarizing features; set to None if already binary."})

    algorithms.append({
        "key": "naive-bayes-bernoulli",
        "display_name": "Naive Bayes (Bernoulli)",
        "family": AlgorithmFamily.CLASSIFICATION,
        "sklearn_class_path": "sklearn.naive_bayes.BernoulliNB",
        "supported_tasks": ["CLASSIFICATION"],
        "default_hyperparameters": {"alpha": 1.0, "fit_prior": True, "binarize": 0.0},
        "hyperparameter_schema": bnb_schema,
        "educational_summary": "Bernoulli Naive Bayes models binary features (presence/absence), often used for text with binary bag-of-words.",
        "educational_details_md": (
            "### Binary likelihood\n"
            "Each feature is treated as a Bernoulli random variable (0/1). "
            "This is useful when presence/absence matters more than frequency.\n"
        ),
        "prerequisites_md": "Bayes rule and binary variables.",
        "typical_use_cases_md": "Text classification with binary indicators, click/no-click features.",
        "strengths_md": "Simple, fast, effective for certain sparse binary domains.",
        "limitations_md": "Less suitable when frequency carries important information.",
        "references": [
            {"title": "scikit-learn: BernoulliNB documentation", "author": "scikit-learn developers", "year": 2025},
        ],
        "sections": [
            ("Binarization", 0, "Features may be thresholded to 0/1, matching the Bernoulli assumption."),
        ]
    })

    # 5) SVM (SVC & SVR)
    svc_schema = _json_schema_base("SVC Hyperparameters", "Hyperparameters for sklearn.svm.SVC (classification).")
    _add_prop(svc_schema, "C", {"type": "number", "exclusiveMinimum": 0.0, "default": 1.0, "description": "Regularization strength; larger C fits training data more closely."})
    _add_prop(svc_schema, "kernel", {"type": "string", "enum": ["linear", "poly", "rbf", "sigmoid"], "default": "rbf"})
    _add_prop(svc_schema, "degree", {"type": "integer", "minimum": 1, "default": 3, "description": "Polynomial degree (if kernel='poly')."})
    _add_prop(svc_schema, "gamma", {"anyOf": [{"type": "string", "enum": ["scale", "auto"]}, {"type": "number", "exclusiveMinimum": 0.0}], "default": "scale"})
    _add_prop(svc_schema, "coef0", {"type": "number", "default": 0.0, "description": "Independent term in poly/sigmoid kernels."})
    _add_prop(svc_schema, "probability", {"type": "boolean", "default": False, "description": "Enable probability estimates (adds cross-validation internally; slower)."})
    _add_prop(svc_schema, "class_weight", {"anyOf": [{"type": "string", "enum": ["balanced"]}, {"type": "null"}], "default": None})
    _add_prop(svc_schema, "max_iter", {"type": "integer", "minimum": -1, "default": -1, "description": "-1 means no limit."})
    _add_prop(svc_schema, "tol", {"type": "number", "exclusiveMinimum": 0.0, "default": 1e-3})

    algorithms.append({
        "key": "svm-classifier",
        "display_name": "Support Vector Machine (SVC)",
        "family": AlgorithmFamily.CLASSIFICATION,
        "sklearn_class_path": "sklearn.svm.SVC",
        "supported_tasks": ["CLASSIFICATION"],
        "default_hyperparameters": {"C": 1.0, "kernel": "rbf", "degree": 3, "gamma": "scale", "coef0": 0.0, "probability": False, "class_weight": None, "max_iter": -1, "tol": 1e-3},
        "hyperparameter_schema": svc_schema,
        "educational_summary": "SVM finds a decision boundary that maximizes the margin between classes; kernels allow non-linear boundaries.",
        "educational_details_md": (
            "### Margin maximization\n"
            "SVM chooses a hyperplane that separates classes with the **largest margin**. "
            "Only some points (support vectors) define the boundary.\n"
            "\n"
            "### Kernels\n"
            "Kernels implicitly map inputs to higher-dimensional spaces:\n"
            "- Linear: straight boundary\n"
            "- RBF: flexible non-linear boundary\n"
            "- Polynomial / Sigmoid: other forms of nonlinearity\n"
            "\n"
            "### Role of C\n"
            "Higher **C** penalizes misclassifications more, potentially reducing bias but increasing variance.\n"
        ),
        "prerequisites_md": "Geometry intuition (hyperplanes), basic optimization intuition.",
        "typical_use_cases_md": "Medium-sized classification problems, high-dimensional spaces, strong baselines.",
        "strengths_md": "Strong performance; flexible with kernels; effective in high dimensions.",
        "limitations_md": "Can be slow on large datasets; hyperparameter tuning important.",
        "references": [
            {"title": "The Nature of Statistical Learning Theory", "author": "Vapnik", "year": 1995},
            {"title": "scikit-learn: SVC documentation", "author": "scikit-learn developers", "year": 2025},
        ],
        "sections": [
            ("Support Vectors", 0, "Only a subset of points determines the boundary."),
            ("Kernel Trick", 1, "Compute similarity without explicitly mapping to high dimensions."),
        ]
    })

    svr_schema = _json_schema_base("SVR Hyperparameters", "Hyperparameters for sklearn.svm.SVR (regression).")
    _add_prop(svr_schema, "C", {"type": "number", "exclusiveMinimum": 0.0, "default": 1.0})
    _add_prop(svr_schema, "kernel", {"type": "string", "enum": ["linear", "poly", "rbf", "sigmoid"], "default": "rbf"})
    _add_prop(svr_schema, "degree", {"type": "integer", "minimum": 1, "default": 3})
    _add_prop(svr_schema, "gamma", {"anyOf": [{"type": "string", "enum": ["scale", "auto"]}, {"type": "number", "exclusiveMinimum": 0.0}], "default": "scale"})
    _add_prop(svr_schema, "coef0", {"type": "number", "default": 0.0})
    _add_prop(svr_schema, "epsilon", {"type": "number", "minimum": 0.0, "default": 0.1, "description": "Epsilon-tube width: errors within epsilon are ignored."})
    _add_prop(svr_schema, "tol", {"type": "number", "exclusiveMinimum": 0.0, "default": 1e-3})
    _add_prop(svr_schema, "max_iter", {"type": "integer", "minimum": -1, "default": -1})

    algorithms.append({
        "key": "svm-regressor",
        "display_name": "Support Vector Regression (SVR)",
        "family": AlgorithmFamily.REGRESSION,
        "sklearn_class_path": "sklearn.svm.SVR",
        "supported_tasks": ["REGRESSION"],
        "default_hyperparameters": {"C": 1.0, "kernel": "rbf", "degree": 3, "gamma": "scale", "coef0": 0.0, "epsilon": 0.1, "tol": 1e-3, "max_iter": -1},
        "hyperparameter_schema": svr_schema,
        "educational_summary": "SVR fits a function while ignoring small errors within an epsilon band; kernels allow non-linear regression.",
        "educational_details_md": (
            "### Epsilon-insensitive loss\n"
            "SVR ignores errors smaller than $\\epsilon$ (creates an **epsilon tube** around the prediction function).\n"
            "\n"
            "### Tradeoffs\n"
            "- Larger **C** fits training data more.\n"
            "- Larger **epsilon** tolerates more deviation (smoother fit).\n"
        ),
        "prerequisites_md": "Regression basics and idea of margins.",
        "typical_use_cases_md": "Non-linear regression for small/medium datasets.",
        "strengths_md": "Flexible, good generalization with proper tuning.",
        "limitations_md": "Scaling to large datasets can be expensive.",
        "references": [
            {"title": "scikit-learn: SVR documentation", "author": "scikit-learn developers", "year": 2025},
        ],
        "sections": [
            ("Epsilon Tube", 0, "Points inside the tube do not contribute to loss."),
        ]
    })

    # 6) Random Forest (Classifier & Regressor)
    rf_schema = _json_schema_base("RandomForest Hyperparameters", "Hyperparameters for sklearn.ensemble.RandomForestClassifier/Regressor.")
    _add_prop(rf_schema, "n_estimators", {"type": "integer", "minimum": 10, "default": 200, "description": "Number of trees in the forest."})
    _add_prop(rf_schema, "max_depth", {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}], "default": None})
    _add_prop(rf_schema, "min_samples_split", {"type": "integer", "minimum": 2, "default": 2})
    _add_prop(rf_schema, "min_samples_leaf", {"type": "integer", "minimum": 1, "default": 1})
    _add_prop(rf_schema, "max_features", {"anyOf": [{"type": "string", "enum": ["sqrt", "log2"]}, {"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0}, {"type": "integer", "minimum": 1}, {"type": "null"}],
                                          "default": "sqrt",
                                          "description": "Number of features to consider for best split (classification default commonly 'sqrt')."})
    _add_prop(rf_schema, "bootstrap", {"type": "boolean", "default": True})
    _add_prop(rf_schema, "n_jobs", {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}], "default": None})
    _add_prop(rf_schema, "random_state", {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": 42})

    algorithms.append({
        "key": "random-forest-classifier",
        "display_name": "Random Forest (Classifier)",
        "family": AlgorithmFamily.CLASSIFICATION,
        "sklearn_class_path": "sklearn.ensemble.RandomForestClassifier",
        "supported_tasks": ["CLASSIFICATION"],
        "default_hyperparameters": {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "bootstrap": True, "n_jobs": None, "random_state": 42},
        "hyperparameter_schema": rf_schema,
        "educational_summary": "Random Forest builds many decision trees on bootstrapped data and averages their outputs to reduce overfitting.",
        "educational_details_md": (
            "### Bagging idea\n"
            "Each tree is trained on a bootstrap sample (random sampling with replacement).\n"
            "\n"
            "### Feature randomness\n"
            "At each split, only a random subset of features is considered. This decorrelates trees.\n"
            "\n"
            "### Why it helps\n"
            "Averaging many high-variance trees reduces variance while keeping bias moderate.\n"
        ),
        "prerequisites_md": "Decision trees basics; idea of averaging to reduce noise.",
        "typical_use_cases_md": "General-purpose strong classifier with minimal tuning.",
        "strengths_md": "Strong accuracy, robust to noise, handles non-linearities.",
        "limitations_md": "Less interpretable than a single tree; larger models.",
        "references": [
            {"title": "Random Forests", "author": "Leo Breiman", "year": 2001},
            {"title": "scikit-learn: RandomForestClassifier documentation", "author": "scikit-learn developers", "year": 2025},
        ],
        "sections": [
            ("Bootstrap Sampling", 0, "Train each tree on a slightly different dataset."),
            ("Ensemble Effect", 1, "Averaging reduces variance and improves stability."),
        ]
    })

    algorithms.append({
        "key": "random-forest-regressor",
        "display_name": "Random Forest (Regressor)",
        "family": AlgorithmFamily.REGRESSION,
        "sklearn_class_path": "sklearn.ensemble.RandomForestRegressor",
        "supported_tasks": ["REGRESSION"],
        "default_hyperparameters": {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": 1.0, "bootstrap": True, "n_jobs": None, "random_state": 42},
        "hyperparameter_schema": rf_schema,
        "educational_summary": "Random Forest regression averages predictions of many decision trees, producing robust non-linear regression.",
        "educational_details_md": (
            "### Regression ensemble\n"
            "For an input $x$, the forest predicts:\n"
            "\n"
            "$$\\hat{y}(x)=\\frac{1}{T}\\sum_{t=1}^{T} \\hat{y}_t(x)$$\n"
            "\n"
            "where each $\\hat{y}_t$ is a tree predictor.\n"
        ),
        "prerequisites_md": "Regression basics and decision trees intuition.",
        "typical_use_cases_md": "Strong non-linear regression baseline with minimal feature engineering.",
        "strengths_md": "Robust, works well out-of-the-box on many tabular datasets.",
        "limitations_md": "Large memory footprint; predictions are not smooth (piecewise).",
        "references": [
            {"title": "Random Forests", "author": "Leo Breiman", "year": 2001},
            {"title": "scikit-learn: RandomForestRegressor documentation", "author": "scikit-learn developers", "year": 2025},
        ],
        "sections": [
            ("Averaging", 0, "Averaging multiple trees stabilizes predictions."),
        ]
    })

    # 7) KNN (Classifier & Regressor)
    knn_schema = _json_schema_base("KNN Hyperparameters", "Hyperparameters for sklearn.neighbors.KNeighborsClassifier/Regressor.")
    _add_prop(knn_schema, "n_neighbors", {"type": "integer", "minimum": 1, "default": 5})
    _add_prop(knn_schema, "weights", {"type": "string", "enum": ["uniform", "distance"], "default": "uniform"})
    _add_prop(knn_schema, "algorithm", {"type": "string", "enum": ["auto", "ball_tree", "kd_tree", "brute"], "default": "auto"})
    _add_prop(knn_schema, "leaf_size", {"type": "integer", "minimum": 1, "default": 30})
    _add_prop(knn_schema, "p", {"type": "integer", "minimum": 1, "default": 2, "description": "Power parameter for Minkowski distance: p=2 (Euclidean), p=1 (Manhattan)."})
    _add_prop(knn_schema, "metric", {"type": "string", "default": "minkowski", "description": "Distance metric; common: 'minkowski', 'euclidean', 'manhattan'."})

    algorithms.append({
        "key": "knn-classifier",
        "display_name": "k-Nearest Neighbors (Classifier)",
        "family": AlgorithmFamily.CLASSIFICATION,
        "sklearn_class_path": "sklearn.neighbors.KNeighborsClassifier",
        "supported_tasks": ["CLASSIFICATION"],
        "default_hyperparameters": {"n_neighbors": 5, "weights": "uniform", "algorithm": "auto", "leaf_size": 30, "p": 2, "metric": "minkowski"},
        "hyperparameter_schema": knn_schema,
        "educational_summary": "k-NN classifies a point by looking at the classes of its nearest neighbors in feature space.",
        "educational_details_md": (
            "### Instance-based learning\n"
            "k-NN does not learn a parametric model. It stores training data and predicts using distances.\n"
            "\n"
            "### Prediction rule\n"
            "For classification, take a majority vote among the **k closest** training points.\n"
            "\n"
            "### Key practical issue: scaling\n"
            "Because distances depend on scale, **standardization** is usually critical.\n"
        ),
        "prerequisites_md": "Distance concept (Euclidean/Manhattan), basic classification.",
        "typical_use_cases_md": "Simple baseline, non-linear boundaries, small/medium datasets.",
        "strengths_md": "Simple, non-linear, no training time (but prediction can be slow).",
        "limitations_md": "Slow predictions on large data; sensitive to scaling and irrelevant features.",
        "references": [
            {"title": "scikit-learn: KNeighborsClassifier documentation", "author": "scikit-learn developers", "year": 2025},
        ],
        "sections": [
            ("Choosing k", 0, "Small k can overfit; large k can underfit."),
            ("Scaling", 1, "Scale features so distance comparisons are meaningful."),
        ]
    })

    algorithms.append({
        "key": "knn-regressor",
        "display_name": "k-Nearest Neighbors (Regressor)",
        "family": AlgorithmFamily.REGRESSION,
        "sklearn_class_path": "sklearn.neighbors.KNeighborsRegressor",
        "supported_tasks": ["REGRESSION"],
        "default_hyperparameters": {"n_neighbors": 5, "weights": "uniform", "algorithm": "auto", "leaf_size": 30, "p": 2, "metric": "minkowski"},
        "hyperparameter_schema": knn_schema,
        "educational_summary": "k-NN regression predicts by averaging the target values of the k nearest neighbors (optionally distance-weighted).",
        "educational_details_md": (
            "### Prediction rule\n"
            "For regression, k-NN predicts the mean (or distance-weighted mean) of neighbor targets.\n"
            "\n"
            "### Behavior\n"
            "The model is locally adaptive but can be noisy if k is too small.\n"
        ),
        "prerequisites_md": "Regression basics and averages.",
        "typical_use_cases_md": "Non-linear regression baseline for small/medium datasets.",
        "strengths_md": "Flexible, simple, captures local patterns.",
        "limitations_md": "Can be noisy; slow at prediction time; needs scaling.",
        "references": [
            {"title": "scikit-learn: KNeighborsRegressor documentation", "author": "scikit-learn developers", "year": 2025},
        ],
        "sections": [
            ("Distance weighting", 0, "Closer points can be given higher influence using weights='distance'."),
        ]
    })

    # 8) K-Means
    kmeans_schema = _json_schema_base("KMeans Hyperparameters", "Hyperparameters for sklearn.cluster.KMeans (clustering).")
    _add_prop(kmeans_schema, "n_clusters", {"type": "integer", "minimum": 2, "default": 8})
    _add_prop(kmeans_schema, "init", {"type": "string", "enum": ["k-means++", "random"], "default": "k-means++"})
    _add_prop(kmeans_schema, "n_init", {"type": "integer", "minimum": 1, "default": 10, "description": "Number of initializations; best result is kept."})
    _add_prop(kmeans_schema, "max_iter", {"type": "integer", "minimum": 10, "default": 300})
    _add_prop(kmeans_schema, "tol", {"type": "number", "exclusiveMinimum": 0.0, "default": 1e-4})
    _add_prop(kmeans_schema, "algorithm", {"type": "string", "enum": ["lloyd", "elkan"], "default": "lloyd"})
    _add_prop(kmeans_schema, "random_state", {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": 42})

    algorithms.append({
        "key": "k-means",
        "display_name": "K-Means Clustering",
        "family": AlgorithmFamily.CLUSTERING,
        "sklearn_class_path": "sklearn.cluster.KMeans",
        "supported_tasks": ["CLUSTERING"],
        "default_hyperparameters": {"n_clusters": 8, "init": "k-means++", "n_init": 10, "max_iter": 300, "tol": 1e-4, "algorithm": "lloyd", "random_state": 42},
        "hyperparameter_schema": kmeans_schema,
        "educational_summary": "K-Means partitions data into k clusters by iteratively assigning points to nearest centroids and updating centroids.",
        "educational_details_md": (
            "### Objective\n"
            "K-Means minimizes within-cluster squared distances (inertia):\n"
            "\n"
            "$$\\min \\sum_{i=1}^{n} \\|x_i - \\mu_{c(i)}\\|^2$$\n"
            "\n"
            "### Algorithm (Lloyd's)\n"
            "1. Initialize centroids\n"
            "2. Assign each point to nearest centroid\n"
            "3. Update centroids as cluster means\n"
            "4. Repeat until convergence\n"
            "\n"
            "### Notes\n"
            "- Sensitive to initialization → use **k-means++** and multiple restarts.\n"
            "- Works best with roughly spherical clusters and similar scales.\n"
        ),
        "prerequisites_md": "Distance concept and averages.",
        "typical_use_cases_md": "Customer segmentation, grouping similar items, vector clustering.",
        "strengths_md": "Fast and simple; scalable; easy to interpret centroids.",
        "limitations_md": "Needs k chosen; poor for non-spherical clusters; sensitive to scaling/outliers.",
        "references": [
            {"title": "Some methods for classification and analysis of multivariate observations", "author": "MacQueen", "year": 1967},
            {"title": "scikit-learn: KMeans documentation", "author": "scikit-learn developers", "year": 2025},
        ],
        "sections": [
            ("Inertia", 0, "Inertia measures compactness: lower is tighter clusters."),
            ("Choosing k", 1, "Use elbow method or silhouette score to select k."),
        ]
    })

    # 9) Neural Networks (MLPClassifier/MLPRegressor)
    mlp_schema = _json_schema_base("MLP Hyperparameters", "Hyperparameters for sklearn.neural_network.MLPClassifier/MLPRegressor.")
    _add_prop(mlp_schema, "hidden_layer_sizes", {
        "type": "array",
        "items": {"type": "integer", "minimum": 1},
        "minItems": 1,
        "default": [100],
        "description": "Sizes of hidden layers, e.g., [100] or [64, 32]."
    })
    _add_prop(mlp_schema, "activation", {"type": "string", "enum": ["relu", "tanh", "logistic", "identity"], "default": "relu"})
    _add_prop(mlp_schema, "solver", {"type": "string", "enum": ["adam", "sgd", "lbfgs"], "default": "adam"})
    _add_prop(mlp_schema, "alpha", {"type": "number", "minimum": 0.0, "default": 0.0001, "description": "L2 regularization term."})
    _add_prop(mlp_schema, "batch_size", {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "string", "enum": ["auto"]}], "default": "auto"})
    _add_prop(mlp_schema, "learning_rate", {"type": "string", "enum": ["constant", "invscaling", "adaptive"], "default": "constant"})
    _add_prop(mlp_schema, "learning_rate_init", {"type": "number", "exclusiveMinimum": 0.0, "default": 0.001})
    _add_prop(mlp_schema, "max_iter", {"type": "integer", "minimum": 50, "default": 300})
    _add_prop(mlp_schema, "early_stopping", {"type": "boolean", "default": False})
    _add_prop(mlp_schema, "validation_fraction", {"type": "number", "exclusiveMinimum": 0.0, "maximum": 0.5, "default": 0.1})
    _add_prop(mlp_schema, "n_iter_no_change", {"type": "integer", "minimum": 1, "default": 10})
    _add_prop(mlp_schema, "random_state", {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": 42})

    algorithms.append({
        "key": "mlp-classifier",
        "display_name": "Neural Network (MLPClassifier)",
        "family": AlgorithmFamily.NEURAL_NETWORK,
        "sklearn_class_path": "sklearn.neural_network.MLPClassifier",
        "supported_tasks": ["CLASSIFICATION"],
        "default_hyperparameters": {
            "hidden_layer_sizes": [100],
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "batch_size": "auto",
            "learning_rate": "constant",
            "learning_rate_init": 0.001,
            "max_iter": 300,
            "early_stopping": False,
            "validation_fraction": 0.1,
            "n_iter_no_change": 10,
            "random_state": 42,
        },
        "hyperparameter_schema": mlp_schema,
        "educational_summary": "An MLP is a feed-forward neural network that learns non-linear functions via layers of neurons and gradient-based optimization.",
        "educational_details_md": (
            "### Architecture\n"
            "An MLP stacks layers of linear transformations followed by non-linear activations:\n"
            "\n"
            "$$\\mathbf{h}^{(1)} = \\phi(\\mathbf{W}^{(1)}\\mathbf{x}+\\mathbf{b}^{(1)}),\\;\\; "
            "\\mathbf{h}^{(2)} = \\phi(\\mathbf{W}^{(2)}\\mathbf{h}^{(1)}+\\mathbf{b}^{(2)}),\\; ...$$\n"
            "\n"
            "### Training\n"
            "Weights are learned by minimizing a loss (e.g., cross-entropy) using **backpropagation** and an optimizer "
            "like Adam/SGD.\n"
            "\n"
            "### Regularization\n"
            "- **alpha (L2)** reduces overfitting by discouraging large weights.\n"
            "- **early_stopping** halts training when validation score stops improving.\n"
        ),
        "prerequisites_md": "Vectors, functions, and the idea of iterative optimization (gradient descent intuition).",
        "typical_use_cases_md": "Non-linear classification on tabular data when simpler models underfit.",
        "strengths_md": "Can model complex non-linear relationships.",
        "limitations_md": "Needs tuning; sensitive to scaling; can overfit; less interpretable.",
        "references": [
            {"title": "Deep Learning", "author": "Goodfellow, Bengio, Courville", "year": 2016},
            {"title": "scikit-learn: MLPClassifier documentation", "author": "scikit-learn developers", "year": 2025},
        ],
        "sections": [
            ("Hidden Layers", 0, "More layers/neurons increase capacity but can increase overfitting."),
            ("Scaling", 1, "MLPs usually require standardization for stable training."),
        ]
    })

    algorithms.append({
        "key": "mlp-regressor",
        "display_name": "Neural Network (MLPRegressor)",
        "family": AlgorithmFamily.NEURAL_NETWORK,
        "sklearn_class_path": "sklearn.neural_network.MLPRegressor",
        "supported_tasks": ["REGRESSION"],
        "default_hyperparameters": {
            "hidden_layer_sizes": [100],
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "batch_size": "auto",
            "learning_rate": "constant",
            "learning_rate_init": 0.001,
            "max_iter": 300,
            "early_stopping": False,
            "validation_fraction": 0.1,
            "n_iter_no_change": 10,
            "random_state": 42,
        },
        "hyperparameter_schema": mlp_schema,
        "educational_summary": "MLPRegressor is an MLP trained to predict continuous targets by minimizing a regression loss.",
        "educational_details_md": (
            "### Regression version\n"
            "The network outputs a real value. Training minimizes a regression loss such as squared error.\n"
            "\n"
            "### Practical tips\n"
            "- Standardize inputs (and often target).\n"
            "- Use early stopping for better generalization.\n"
        ),
        "prerequisites_md": "Regression basics and neural network intuition.",
        "typical_use_cases_md": "Non-linear regression on tabular data when tree ensembles are not preferred.",
        "strengths_md": "Flexible function approximation.",
        "limitations_md": "Tuning is sensitive; not as strong as tree ensembles on many tabular datasets.",
        "references": [
            {"title": "scikit-learn: MLPRegressor documentation", "author": "scikit-learn developers", "year": 2025},
        ],
        "sections": [
            ("Overfitting Control", 0, "Use alpha and early stopping to reduce overfitting."),
        ]
    })

    # Write to DB (idempotent)
    for a in algorithms:
        obj, was_created = Algorithm.objects.update_or_create(
            key=a["key"],
            defaults={
                "display_name": a["display_name"],
                "family": a["family"],
                "sklearn_class_path": a["sklearn_class_path"],
                "supported_tasks": a["supported_tasks"],
                "default_hyperparameters": a["default_hyperparameters"],
                "hyperparameter_schema": a["hyperparameter_schema"],
                "educational_summary": a["educational_summary"],
                "educational_details_md": a["educational_details_md"],
                "prerequisites_md": a["prerequisites_md"],
                "typical_use_cases_md": a["typical_use_cases_md"],
                "strengths_md": a["strengths_md"],
                "limitations_md": a["limitations_md"],
                "references": a["references"],
            }
        )
        if was_created:
            created += 1
        else:
            updated += 1

        # Tutorial sections: replace existing to keep consistent with seed
        AlgorithmTutorialSection.objects.filter(algorithm=obj).delete()
        for title, order, content_md in a.get("sections", []):
            AlgorithmTutorialSection.objects.create(
                algorithm=obj,
                title=title,
                order=order,
                content_md=content_md,
                quiz_questions=[],
            )

    return {"created": created, "updated": updated}


class Command(BaseCommand):
    help = "Seed ML algorithms registry with sklearn class paths, hyperparameter schemas, and educational tutorials."

    def handle(self, *args, **options):
        result = seed_algorithms()
        self.stdout.write(self.style.SUCCESS(
            f"seed_algorithms completed. created={result['created']} updated={result['updated']}"
        ))
