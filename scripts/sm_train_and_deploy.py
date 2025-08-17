import os
import time
import json
import boto3
from datetime import datetime

# AWS clients & settings
region       = os.environ["AWS_REGION"]
sagemaker    = boto3.client("sagemaker", region_name=region)
account_id   = boto3.client("sts").get_caller_identity()["Account"]

# Inputs from GitHub Actions env
TRAIN_IMAGE_URI        = os.environ["TRAIN_IMAGE_URI"]
INFER_IMAGE_URI        = os.environ["INFER_IMAGE_URI"]
ROLE_ARN               = os.environ["SAGEMAKER_EXECUTION_ROLE_ARN"]
S3_BUCKET              = os.environ["S3_BUCKET"]
ENDPOINT_BASE          = os.environ.get("ENDPOINT_NAME_BASE", "urbansound-audio")
FORCE_RETRAIN          = os.environ.get("FORCE_RETRAIN", "false").lower() == "true"
TRAIN_INSTANCE         = os.environ.get("INSTANCE_TYPE", "ml.m5.large")
SERVE_INSTANCE         = os.environ.get("SERVE_INSTANCE_TYPE", "ml.g4dn.xlarge")
ENDPOINT_ENV           = os.environ.get("ENDPOINT_ENV", "staging")
GIT_SHA                = os.environ.get("GIT_SHA", "local")

# S3 prefixes for data and outputs
S3_TRAIN_PREFIX       = f"s3://{S3_BUCKET}/training/"
S3_OUTPUT_PREFIX      = f"s3://{S3_BUCKET}/artifacts"

# Unique job timestamp
ts       = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
job_name = f"urbansound-train-{ts}"

def wait_training(name):
    print(f"Waiting for training job: {name}")
    sm = boto3.client("sagemaker", region_name=region)
    while True:
        desc   = sm.describe_training_job(TrainingJobName=name)
        status = desc["TrainingJobStatus"]
        print(f"  status={status}")
        if status in ("Completed", "Failed", "Stopped"):
            return desc
        time.sleep(30)

def create_or_update_endpoint(model_name, endpoint_name, instance_type):
    print(f"Creating endpoint config for {endpoint_name}")
    sm = boto3.client("sagemaker", region_name=region)
    config_name = f"{endpoint_name}-cfg-{ts}"

    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName":           "AllTraffic",
            "ModelName":             model_name,
            "InitialInstanceCount":  1,
            "InstanceType":          instance_type,
            "InitialVariantWeight":  1.0
        }],
    )

    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        print(f"Updating endpoint {endpoint_name}...")
        sm.update_endpoint(
            EndpointName=          endpoint_name,
            EndpointConfigName=    config_name
        )
    except sm.exceptions.ClientError:
        print(f"Creating endpoint {endpoint_name}...")
        sm.create_endpoint(
            EndpointName=          endpoint_name,
            EndpointConfigName=    config_name
        )

def main():
    print("Launching SageMaker training...")

    # InputDataConfig pointing at the raw files under /training/
    input_cfg = [{
        "ChannelName": "training",
        "DataSource": {
            "S3DataSource": {
                "S3DataType":             "S3Prefix",
                "S3Uri":                  S3_TRAIN_PREFIX,
                "S3DataDistributionType": "FullyReplicated"
            }
        },
        "InputMode": "File"
    }]

    out_path = f"{S3_OUTPUT_PREFIX}/{job_name}"
    hyperparams = {"EPOCHS": "5"}

    sagemaker.create_training_job(
        TrainingJobName=job_name,
        RoleArn=ROLE_ARN,
        AlgorithmSpecification={
            "TrainingImage":    TRAIN_IMAGE_URI,
            "TrainingInputMode":"File"
        },
        InputDataConfig=input_cfg,
        OutputDataConfig={"S3OutputPath": out_path},
        ResourceConfig={
            "InstanceType":     TRAIN_INSTANCE,
            "InstanceCount":    1,
            "VolumeSizeInGB":   50
        },
        StoppingCondition={"MaxRuntimeInSeconds": 2*3600},
        HyperParameters=hyperparams,
        Tags=[
            {"Key": "project", "Value": "urbansound"},
            {"Key": "commit",  "Value": GIT_SHA},
            {"Key": "env",     "Value": ENDPOINT_ENV}
        ],
        EnableNetworkIsolation=False,
        EnableInterContainerTrafficEncryption=False,
        EnableManagedSpotTraining=False
    )

    desc = wait_training(job_name)
    if desc["TrainingJobStatus"] != "Completed":
        raise RuntimeError(f"Training failed: {desc.get('FailureReason')}")

    model_artifact = desc["ModelArtifacts"]["S3ModelArtifacts"]
    print(f"Model artifact: {model_artifact}")

    # Create SageMaker Model for inference
    model_name = f"{ENDPOINT_BASE}-mdl-{ts}"
    primary_container = {
        "Image":        INFER_IMAGE_URI,
        "Mode":         "SingleModel",
        "ModelDataUrl": model_artifact,
        "Environment": {
            "SAGEMAKER_PROGRAM": "serve.py"
        }
    }

    sagemaker.create_model(
        ModelName=         model_name,
        ExecutionRoleArn=  ROLE_ARN,
        PrimaryContainer=  primary_container
    )

    endpoint_name = f"{ENDPOINT_BASE}-{ENDPOINT_ENV}"
    create_or_update_endpoint(model_name, endpoint_name, SERVE_INSTANCE)

    print("All done.")
    print(json.dumps({
        "endpoint_name": endpoint_name,
        "model_name":    model_name,
        "artifact":      model_artifact
    }, indent=2))

if __name__ == "__main__":
    main()
