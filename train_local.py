import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import logging

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score
from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.filterwarnings('ignore')

from utils import update_model, save_simple_metrics_report, plot_model_performance

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

logger.info('Loading local data...')
df = pd.read_csv('data/data.csv')
df.drop('id', axis=1, inplace=True)
df.drop(['concave_points_mean', 'radius_worst', 'perimeter_worst', 'area_worst'], axis=1, inplace=True)
try:
    df.drop('Unnamed: 32', axis=1, inplace=True)
except Exception:
    pass

sc = MinMaxScaler()
X = df.drop('diagnosis', axis=1)
y = df.diagnosis

sampler = RandomUnderSampler(random_state=42)
X, y = sampler.fit_resample(X, y)

label_map = {label: i for i, label in enumerate(sorted(y.unique()))}
y = y.map(label_map).astype(int)
logger.info(f'Label mapping: {label_map}')
logger.info(f'Column order (positional, for inference): {X.columns.tolist()}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train = pd.DataFrame(sc.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(sc.transform(X_test), columns=X.columns)

# Use positional indices instead of column names, so this also works on plain
# arrays/lists (e.g. Vertex AI's prebuilt predictor), not just DataFrames.
column_transformer = ColumnTransformer([
    ("scaler", sc, list(range(len(X.columns))))
], remainder="passthrough")

model = Pipeline([
    ('datafeed', column_transformer),
    ('selector', SelectKBest(f_classif, k='all')),
    ('classifier', LogisticRegression(penalty='l2', tol=1e-20, C=0.9, verbose=0, n_jobs=-1, max_iter=1000))
])

param_grid = {
    'classifier__solver': ['liblinear', 'lbfgs']
}

grid = GridSearchCV(model, param_grid, verbose=1, cv=5, scoring='f1')
logger.info('Training...')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

results = cross_validate(model, X, y, return_train_score=True, cv=5)
train_score = np.round(np.mean(results['train_score']), 2)
test_score = np.round(np.mean(results['test_score']), 2)
model_f1 = f1_score(y_pred, y_test, pos_label=1)
model_acc = accuracy_score(y_pred, y_test)

logger.info(f'F1: {model_f1:.3f}, Accuracy: {model_acc:.3f}')

logger.info('Saving model to models/model.pkl ...')
update_model(best_model)
save_simple_metrics_report(train_score, test_score, model_f1, best_model)
plot_model_performance(y_pred, y_test)

logger.info('Done.')
