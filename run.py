from sklearn import svm
from sklearn import datasets
import mlflow

# import the IrisClassifier class defined above
from iris_classifier import IrisClassifier

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

model_uri = f"runs:/{run.info.run_id}/model"
print(f"Retrieving model with uri={model_uri}")
mlflow_loaded_model = mlflow.sklearn.load_model(model_uri)

# Create a iris classifier service instance
iris_classifier_service = IrisClassifier()

# Pack the newly trained model artifact
iris_classifier_service.pack("model", mlflow_loaded_model)

# Save the prediction service to disk for model serving
saved_path = iris_classifier_service.save()
