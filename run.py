from sklearn import svm
from sklearn import datasets
import mlflow

# Load training data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Model Training and saving in MLflow
clf = svm.SVC(gamma="scale")
with mlflow.start_run() as run:
    clf.fit(X, y)
    mlflow.sklearn.log_model(
        sk_model=clf, artifact_path="model", signature=mlflow.models.signature.infer_signature(X),
    )
