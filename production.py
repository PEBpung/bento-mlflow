# MLflow client 생성
import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri=mlflow_endpoint)

# 모델 이름 정의
model_name="{model_name}"

# 모델 repository 검색 및 조회
filter_string = "name='{}'".format(model_name)
results = client.search_model_versions(filter_string)

for res in results:
    print("name={}; run_id={}; version={}; current_stage={}".format(res.name, res.run_id, res.version, res.current_stage))

# Production stage 모델 버전 선택
for res in results:
    if res.current_stage == "Production":
        deploy_version = res.version

# MLflow production 모델 버전 다운로드 URI 획득
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
model_uri = client.get_model_version_download_uri(model_name, deploy_version)

# 모델 다운로드
download_path = "{local_download_path}"
mlflow_run_id = "{run_id}"
mlflow_run_id_artifacts_name = "{artifacts_model_name}"

client.download_artifacts(mlflow_run_id, mlflow_run_id_artifacts_name, dst_path=download_path)

# 다운로드 모델 load & predict 예시
reconstructed_model = mlflow.{framework}.load_model("{download_path}/{model_name}".format(download_path=download_path,model_name=mlflow_run_id_artifacts_name))
output = reconstructed_model.predict(input_feature)