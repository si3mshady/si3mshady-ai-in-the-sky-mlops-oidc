import os
import time
import json
import boto3
from datetime import datetime

# SageMaker client
region      = os.environ["AWS_REGION"]
sagemaker   = boto3.client("sagemaker", region_name=region)

# Image URIs
TRAIN_IMAGE = os.environ["TRAIN_IMAGE_URI"]
INFER_IMAGE = os.environ["INFER_IMAGE_URI"]
ROLE_ARN    = os.environ["SAGEMAKER_EXECUTION_ROLE_ARN"]

# Real data location
S3_BUCKET   = os.environ["S3_BUCKET"]
S3_PREFIX   = os.environ.get("S3_TRAIN_PREFIX", "training/").rstrip("/") + "/"

# Endpoint settings
ENDPOINT_BASE = os.environ.get("ENDPOINT_NAME_BASE", "urbansound-audio")
ENDPOINT_ENV  = os.environ.get("ENDPOINT_ENV", "production")
GIT_SHA       = os.environ.get("GIT_SHA", "local")

# Resources & hyperparameters
TRAIN_INSTANCE = os.environ.get("INSTANCE_TYPE", "ml.m5.large")
SERVE_INSTANCE = os.environ.get("SERVE_INSTANCE_TYPE", "ml.g4dn.xlarge")
EPOCHS         = os.environ.get("EPOCHS", "5")

# Job naming
ts        = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
job_name  = f"urbansound-train-{ts}"
out_path  = f"s3://{S3_BUCKET}/artifacts/{job_name}"

def wait_training(name):
    sm = boto3.client("sagemaker", region_name=region)
    while True:
        desc = sm.describe_training_job(TrainingJobName=name)
        status = desc["TrainingJobStatus"]
        print(f"  status={status}")
        if status in ("Completed","Failed","Stopped"):
            return desc
        time.sleep(30)

def create_or_update_endpoint(model_name, endpoint_name, instance_type):
    sm = boto3.client("sagemaker", region_name=region)
    cfg_name = f"{endpoint_name}-cfg-{ts}"
    sm.create_endpoint_config(
        EndpointConfigName=cfg_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName":    model_name,
            "InstanceType": instance_type,
            "InitialInstanceCount": 1,
            "InitialVariantWeight": 1.0
        }],
    )
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=cfg_name)
    except sm.exceptions.ClientError:
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=cfg_name)

def main():
    print("Starting SageMaker training:", job_name)

    # MUST point at your real data prefix so SageMaker sees objects[1]
    input_cfg = [{
        "ChannelName": "training",
        "DataSource": {
            "S3DataSource": {
                "S3DataType":             "S3Prefix",
                "S3Uri":                  f"s3://{S3_BUCKET}/{S3_PREFIX}",
                "S3DataDistributionType": "FullyReplicated"
            }
        },
        "InputMode": "File"
    }]

    sagemaker.create_training_job(
        TrainingJobName=job_name,
        RoleArn=ROLE_ARN,
        AlgorithmSpecification={
            "TrainingImage": TRAIN_IMAGE,
            "TrainingInputMode": "File"
        },
        InputDataConfig=input_cfg,                # now valid
        OutputDataConfig={"S3OutputPath": out_path},
        ResourceConfig={
            "InstanceType": TRAIN_INSTANCE,
            "InstanceCount": 1,
            "VolumeSizeInGB": 50
        },
        StoppingCondition={"MaxRuntimeInSeconds": 2*3600},
        HyperParameters={"EPOCHS": EPOCHS},
        Environment={
            "S3_TRAIN_BUCKET": S3_BUCKET,
            "S3_TRAIN_PREFIX": S3_PREFIX,
            "SM_MODEL_DIR":     "/opt/ml/model"
        },
        Tags=[
            {"Key": "project", "Value": "urbansound"},
            {"Key": "commit",  "Value": GIT_SHA},
            {"Key": "env",     "Value": ENDPOINT_ENV}
        ]
    )

    desc = wait_training(job_name)
    if desc["TrainingJobStatus"] != "Completed":
        raise RuntimeError(f"Training failed: {desc.get('FailureReason')}")

    artifact = desc["ModelArtifacts"]["S3ModelArtifacts"]
    print("Model artifact:", artifact)

    # Deploy model
    model_name = f"{ENDPOINT_BASE}-mdl-{ts}"
    container  = {
        "Image":        INFER_IMAGE,
        "Mode":         "SingleModel",
        "ModelDataUrl": artifact,
        "Environment":  {"SAGEMAKER_PROGRAM":"serve.py"}
    }
    sagemaker.create_model(ModelName=model_name, ExecutionRoleArn=ROLE_ARN, PrimaryContainer=container)
    endpoint = f"{ENDPOINT_BASE}-{ENDPOINT_ENV}"
    create_or_update_endpoint(model_name, endpoint, SERVE_INSTANCE)

    print("Endpoint live at:", endpoint)

if __name__=="__main__":
    main()
