import dagshub
import mlflow 

mlflow.set_tacking_uri("https://dagshub.com/vinayak910/mlops-mini-project.mlflow")
dagshub.init(repo_owner='vinayak910', repo_name='mlops-mini-project', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)