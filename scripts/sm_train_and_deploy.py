import os
import time
import json
import boto3
from datetime import datetime

region     = os.environ["AWS_REGION"]
sagemaker  = boto3.client("sagemaker", region_name=region)

# GitHub Actions inputs
TRAIN_IMAGE_URI = os.environ["TRAIN_IMAGE_URI"]
ROLE_ARN        = os.environ["SAGEMAKER_EXECUTION_ROLE_ARN"]
S3_BUCKET       = os.environ["S3_BUCKET"]
S3_PREFIX       = os.environ.get("S3_PREFIX", "training/")

# Other configs
TRAIN_INSTANCE = os.environ.get("INSTANCE_TYPE", "ml.m5.large")
HYPERPARAMS    = {"EPOCHS": os.environ.get("EPOCHS", "5")}

ts       = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
job_name = f"urbansound-train-{ts}"
out_path = f"s3://{S3_BUCKET}/artifacts/{job_name}"

def wait_training(name):
    sm = boto3.client("sagemaker", region_name=region)
    while True:
        desc = sm.describe_training_job(TrainingJobName=name)
        status = desc["TrainingJobStatus"]
        print("status=", status)
        if status in ("Completed", "Failed", "Stopped"):
            return desc
        time.sleep(30)

def main():
    print("Launching SageMaker training...")

    # Create training job, but do NOT configure S3 channels
    sagemaker.create_training_job(
        TrainingJobName=job_name,
        RoleArn=ROLE_ARN,
        AlgorithmSpecification={
            "TrainingImage": TRAIN_IMAGE_URI,
            "TrainingInputMode": "File"
        },
        InputDataConfig=[],  # no channels
        OutputDataConfig={"S3OutputPath": out_path},
        ResourceConfig={
            "InstanceType": TRAIN_INSTANCE,
            "InstanceCount": 1,
            "VolumeSizeInGB": 50
        },
        StoppingCondition={"MaxRuntimeInSeconds": 2*3600},
        HyperParameters=HYPERPARAMS,
        Environment={
            # Tell the container where to fetch data
            "S3_TRAIN_BUCKET": S3_BUCKET,
            "S3_TRAIN_PREFIX": S3_PREFIX,
            "SM_MODEL_DIR": "/opt/ml/model",
            "SM_OUTPUT_DATA_DIR": "/opt/ml/output"
        }
    )

    desc = wait_training(job_name)
    if desc["TrainingJobStatus"] != "Completed":
        raise RuntimeError(f"Training failed: {desc.get('FailureReason')}")

    print(json.dumps({
        "ModelArtifacts": desc["ModelArtifacts"],
        "TrainingJobArn": desc["TrainingJobArn"]
    }, indent=2))

if __name__ == "__main__":
    main()
