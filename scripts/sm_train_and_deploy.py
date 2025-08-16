# scripts/sm_train_and_deploy.py
import os, time, json, boto3
from datetime import datetime

region      = os.environ["AWS_REGION"]
train_image = os.environ["TRAIN_IMAGE_URI"]
infer_image = os.environ["INFER_IMAGE_URI"]
bucket      = os.environ["S3_BUCKET"]
role_arn    = os.environ["SAGEMAKER_EXECUTION_ROLE_ARN"]
endpoint_base = os.environ.get("ENDPOINT_NAME_BASE", "urbansound-audio")
force_retrain = os.environ.get("FORCE_RETRAIN","false").lower()=="true"
train_instance= os.environ.get("INSTANCE_TYPE","ml.m5.large")
serve_instance= os.environ.get("SERVE_INSTANCE_TYPE","ml.g4dn.xlarge")
endpoint_env  = os.environ.get("ENDPOINT_ENV","staging")
git_sha       = os.environ.get("GIT_SHA","local")

sm = boto3.client("sagemaker", region_name=region)

job_time = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
job_name = f"urbansound-train-{job_time}"
s3_output = f"s3://{bucket}/model-artifacts/{job_name}"
endpoint_name = f"{endpoint_base}-{endpoint_env}"

def latest_completed(prefix="urbansound-train-"):
    resp = sm.list_training_jobs(MaxResults=10, SortBy="CreationTime", SortOrder="Descending",
                                 StatusEquals="Completed", NameContains=prefix)
    jobs = resp.get("TrainingJobSummaries") or []
    return jobs[0]["TrainingJobName"] if jobs else None

def artifacts_of(name):
    return sm.describe_training_job(TrainingJobName=name)["ModelArtifacts"]["S3ModelArtifacts"]

def model_exists(name):
    try:
        sm.describe_model(ModelName=name)
        return True
    except sm.exceptions.ResourceNotFound:
        return False

reuse = False
if not force_retrain:
    last = latest_completed()
    if last:
        model_artifacts_s3 = artifacts_of(last)
        print(f"INFO: Reusing training job {last}: {model_artifacts_s3}")
        reuse = True

if not reuse:
    print(f"🚀 Starting training: {job_name}")
    sm.create_training_job(
        TrainingJobName=job_name,
        HyperParameters={"epochs": "8"},
        AlgorithmSpecification={"TrainingImage": train_image, "TrainingInputMode": "File"},
        RoleArn=role_arn,
        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{bucket}/data/UrbanSound.tar.gz",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/x-tar",
                "CompressionType": "None",
            }
        ],
        OutputDataConfig={"S3OutputPath": s3_output},
        ResourceConfig={"InstanceType": train_instance, "InstanceCount": 1, "VolumeSizeInGB": 50},
        StoppingCondition={"MaxRuntimeInSeconds": 5400},
        EnableNetworkIsolation=False,
    )

    while True:
        d = sm.describe_training_job(TrainingJobName=job_name)
        st = d["TrainingJobStatus"]
        print(f"⏳ Training status: {st}")
        if st in ("Completed","Failed","Stopped"): break
        time.sleep(30)

    if st != "Completed": 
        raise RuntimeError(f"Training failed with status: {st}")

    model_artifacts_s3 = d["ModelArtifacts"]["S3ModelArtifacts"]

# Create or replace model
model_name = f"{endpoint_base}-model-{endpoint_env}-{git_sha[:8]}"

if model_exists(model_name):
    print(f"🗑️ Deleting existing model: {model_name}")
    sm.delete_model(ModelName=model_name)

print(f"🧱 Creating model: {model_name}")
sm.create_model(
    ModelName=model_name,
    PrimaryContainer={
        "Image": infer_image,
        "Mode": "SingleModel",
        "ModelDataUrl": model_artifacts_s3,
        "Environment": {"SAGEMAKER_PROGRAM":"serve.py","SAGEMAKER_REGION":region},
    },
    ExecutionRoleArn=role_arn,
)

config_name = f"{endpoint_base}-cfg-{endpoint_env}-{git_sha[:8]}"
print(f"⚙️ Creating endpoint config: {config_name}")

try:
    sm.describe_endpoint_config(EndpointConfigName=config_name)
    print(f"🗑️ Deleting existing config: {config_name}")
    sm.delete_endpoint_config(EndpointConfigName=config_name)
except sm.exceptions.ResourceNotFound:
    pass

sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[{
        "VariantName":"AllTraffic","ModelName":model_name,
        "InitialInstanceCount":1,"InstanceType":serve_instance,"InitialVariantWeight":1.0
    }],
)

def endpoint_exists(name):
    try: 
        sm.describe_endpoint(EndpointName=name)
        return True
    except sm.exceptions.ResourceNotFound: 
        return False

if endpoint_exists(endpoint_name):
    print(f"♻️ Updating endpoint: {endpoint_name}")
    sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
else:
    print(f"🚀 Creating endpoint: {endpoint_name}")
    sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)

while True:
    ep = sm.describe_endpoint(EndpointName=endpoint_name)
    st = ep["EndpointStatus"]
    print(f"⏳ Endpoint status: {st}")
    if st in ("InService","Failed"): break
    time.sleep(30)

if st != "InService": 
    raise RuntimeError(f"Endpoint deployment failed with status: {st}")

print("✅ Deployment successful!")
print(json.dumps({
    "endpoint_name": endpoint_name, 
    "artifacts": model_artifacts_s3,
    "model_name": model_name,
    "config_name": config_name
}, indent=2))

