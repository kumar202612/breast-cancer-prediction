"""
Consolidated deploy script for the breast-cancer-prediction project.
Covers: local data generation -> training -> GCS upload -> Vertex AI
Model Registry -> Endpoint deploy/redeploy -> test prediction -> undeploy.

Usage:
    python deploy.py --full                 # run everything, deploy, test, then undeploy
    python deploy.py --train                # regenerate data + retrain only
    python deploy.py --deploy                # upload+register+deploy only (assumes model.pkl exists)
    python deploy.py --test                  # send a test prediction to the current endpoint
    python deploy.py --undeploy              # undeploy + delete the endpoint (stop billing)
"""

import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ---- Config: change these if project/region/bucket ever change ----
PROJECT_ID = "mlops-vertex-ai-tsk1"
REGION = "us-east1"
BUCKET_NAME = "mlops-vertex-ai-tsk1-breastcancer"
BUCKET_URI = f"gs://{BUCKET_NAME}"
MODEL_DISPLAY_NAME = "breast-cancer-lr-model"
ENDPOINT_DISPLAY_NAME = "breast-cancer-endpoint"
SERVING_CONTAINER = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
MACHINE_TYPE = "n1-standard-2"

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(REPO_DIR, "data", "data.csv")
MODEL_PATH = os.path.join(REPO_DIR, "models", "model.pkl")

# Column order used for training AND for building test/inference payloads.
# Must stay in sync with generate_data() + train().
FEATURE_COLUMNS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se",
    "fractal_dimension_se", "texture_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]

SAMPLE_INSTANCE = [
    20.57, 17.77, 132.9, 1326, 0.08474, 0.07864, 0.0869, 0.1812, 0.05667,
    0.5435, 0.7339, 3.398, 74.08, 0.005225, 0.01308, 0.0186, 0.0134, 0.01389, 0.003532,
    0.1238, 0.1238, 0.1866, 0.2416, 0.186, 0.08902, 0.08902,
]


def generate_data():
    """Regenerate data/data.csv from sklearn's built-in Wisconsin dataset
    (the repo's DVC-tracked data isn't accessible to us)."""
    from sklearn.datasets import load_breast_cancer

    logger.info("Generating local dataset...")
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()

    feature_map = {}
    for col in data.feature_names:
        if col.startswith("mean "):
            base, suffix = col[len("mean "):], "_mean"
        elif col.startswith("worst "):
            base, suffix = col[len("worst "):], "_worst"
        elif col.endswith(" error"):
            base, suffix = col[:-len(" error")], "_se"
        else:
            base, suffix = col, ""
        feature_map[col] = base.replace(" ", "_") + suffix
    df.rename(columns=feature_map, inplace=True)

    df["diagnosis"] = df["target"].map({0: "M", 1: "B"})
    df.drop(columns=["target"], inplace=True)
    df.insert(0, "id", range(1, len(df) + 1))
    df["Unnamed: 32"] = None

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    logger.info(f"Wrote {df.shape} to {DATA_PATH}")


def train():
    """Train the LogisticRegression pipeline and save models/model.pkl
    using positional column indices (works with plain arrays, not just
    DataFrames) and plain pickle at protocol 4 (matches the Vertex
    prebuilt sklearn container's plain pickle.load())."""
    from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import f1_score, accuracy_score
    from imblearn.under_sampling import RandomUnderSampler
    import warnings
    warnings.filterwarnings("ignore")

    logger.info("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df.drop("id", axis=1, inplace=True)
    df.drop(["concave_points_mean", "radius_worst", "perimeter_worst", "area_worst"], axis=1, inplace=True)
    try:
        df.drop("Unnamed: 32", axis=1, inplace=True)
    except Exception:
        pass

    assert list(df.drop("diagnosis", axis=1).columns) == FEATURE_COLUMNS, (
        "Column order in data.csv doesn't match FEATURE_COLUMNS — "
        "inference payloads would silently misalign."
    )

    sc = MinMaxScaler()
    X = df.drop("diagnosis", axis=1)
    y = df.diagnosis

    sampler = RandomUnderSampler(random_state=42)
    X, y = sampler.fit_resample(X, y)

    label_map = {label: i for i, label in enumerate(sorted(y.unique()))}
    y = y.map(label_map).astype(int)
    logger.info(f"Label mapping: {label_map}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = pd.DataFrame(sc.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(sc.transform(X_test), columns=X.columns)

    column_transformer = ColumnTransformer(
        [("scaler", sc, list(range(len(X.columns))))], remainder="passthrough"
    )

    model = Pipeline([
        ("datafeed", column_transformer),
        ("selector", SelectKBest(f_classif, k="all")),
        ("classifier", LogisticRegression(
            penalty="l2", tol=1e-20, C=0.9, verbose=0, n_jobs=-1, max_iter=1000
        )),
    ])

    param_grid = {"classifier__solver": ["liblinear", "lbfgs"]}
    grid = GridSearchCV(model, param_grid, verbose=1, cv=5, scoring="f1")

    logger.info("Training...")
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    results = cross_validate(model, X, y, return_train_score=True, cv=5)
    f1 = f1_score(y_pred, y_test, pos_label=1)
    acc = accuracy_score(y_pred, y_test)
    logger.info(f"F1: {f1:.3f}, Accuracy: {acc:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f, protocol=4)
    logger.info(f"Saved model to {MODEL_PATH}")

    # Local sanity check mimicking exactly how Vertex will call it
    with open(MODEL_PATH, "rb") as f:
        reloaded = pickle.load(f)
    pred = reloaded.predict([SAMPLE_INSTANCE])
    logger.info(f"Local sanity-check prediction on sample instance: {pred}")


def upload_to_gcs():
    import subprocess
    logger.info("Uploading model to GCS (bucket contains ONLY model.pkl)...")
    subprocess.run(["gcloud", "storage", "rm", f"{BUCKET_URI}/models/model.pkl"],
                    capture_output=True)  # ok if it doesn't exist yet
    subprocess.run(
        ["gcloud", "storage", "cp", MODEL_PATH, f"{BUCKET_URI}/models/model.pkl"],
        check=True,
    )


def register_and_deploy():
    """Register a new model version and deploy it onto the existing
    endpoint if one exists (reusing it, avoiding duplicate endpoint
    cost); otherwise creates a new endpoint. Ends with only the new
    model receiving 100% traffic — any prior model on the endpoint is
    undeployed."""
    from google.cloud import aiplatform
    import time

    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    version_tag = time.strftime("%Y%m%d-%H%M%S")
    display_name = f"{MODEL_DISPLAY_NAME}-{version_tag}"

    logger.info(f"Registering model as {display_name}...")
    uploaded_model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=f"{BUCKET_URI}/models/",
        serving_container_image_uri=SERVING_CONTAINER,
    )
    logger.info(f"Registered: {uploaded_model.resource_name}")

    # Reuse an existing endpoint with this display name if one exists
    existing = aiplatform.Endpoint.list(
        filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"', order_by="create_time desc"
    )
    if existing:
        endpoint = existing[0]
        logger.info(f"Reusing existing endpoint: {endpoint.resource_name}")
        old_deployed_ids = [dm.id for dm in endpoint.list_models()]
    else:
        endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_DISPLAY_NAME)
        logger.info(f"Created new endpoint: {endpoint.resource_name}")
        old_deployed_ids = []

    logger.info("Deploying new model version (this takes several minutes)...")
    uploaded_model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=display_name,
        machine_type=MACHINE_TYPE,
        min_replica_count=1,
        max_replica_count=1,
        traffic_percentage=100,  # shift all traffic to the new version immediately
    )
    logger.info("Deploy complete.")

    # Clean up any previously deployed model(s) on this endpoint
    for old_id in old_deployed_ids:
        logger.info(f"Undeploying superseded model {old_id}...")
        endpoint.undeploy(deployed_model_id=old_id)

    return endpoint


def test_predict(endpoint=None):
    from google.cloud import aiplatform
    aiplatform.init(project=PROJECT_ID, location=REGION)

    if endpoint is None:
        existing = aiplatform.Endpoint.list(
            filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"', order_by="create_time desc"
        )
        if not existing:
            logger.error("No endpoint found to test.")
            return
        endpoint = existing[0]

    prediction = endpoint.predict(instances=[SAMPLE_INSTANCE])
    label = "Malignant" if prediction.predictions[0] == 1 else "Benign"
    logger.info(f"Prediction: {prediction.predictions[0]} ({label})")


def undeploy_all():
    """Undeploy and delete the endpoint entirely — stops all billing.
    Registered models in Model Registry are untouched (free to keep)."""
    from google.cloud import aiplatform
    aiplatform.init(project=PROJECT_ID, location=REGION)

    existing = aiplatform.Endpoint.list(filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"')
    if not existing:
        logger.info("No endpoint found — nothing to undeploy.")
        return

    for endpoint in existing:
        for dm in endpoint.list_models():
            logger.info(f"Undeploying {dm.id} from {endpoint.resource_name}...")
            endpoint.undeploy(deployed_model_id=dm.id)
        logger.info(f"Deleting endpoint {endpoint.resource_name}...")
        endpoint.delete()
    logger.info("All endpoints undeployed and deleted.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="train + deploy + test + undeploy")
    parser.add_argument("--train", action="store_true", help="regenerate data + retrain only")
    parser.add_argument("--deploy", action="store_true", help="upload + register + deploy only")
    parser.add_argument("--test", action="store_true", help="send a test prediction")
    parser.add_argument("--undeploy", action="store_true", help="undeploy + delete endpoint")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(1)

    if args.full or args.train:
        generate_data()
        train()

    endpoint = None
    if args.full or args.deploy:
        upload_to_gcs()
        endpoint = register_and_deploy()

    if args.full or args.test:
        test_predict(endpoint)

    if args.full:
        input("Press Enter to undeploy and stop billing (or Ctrl+C to leave it running)...")
        undeploy_all()
    elif args.undeploy:
        undeploy_all()


if __name__ == "__main__":
    main()
