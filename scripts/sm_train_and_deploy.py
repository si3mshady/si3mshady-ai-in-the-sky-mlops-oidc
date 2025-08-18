#!/usr/bin/env python3
import os, time, json, argparse
from datetime import datetime
import boto3

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--execution-role", required=True)
    p.add_argument("--train-image-uri", required=True)
    p.add_argument("--infer-image-uri", required=True)
    p.add_argument("--endpoint-base", required=True)
    p.add_argument("--force-retrain", default="false")
    p.add_argument("--train-instance", default="ml.m5.large")
    p.add_argument("--serve-instance", default="ml.g4dn.xlarge")
    p.add_argument("--env", default="staging")
    p.add_argument("--git-sha", default="local")
    p.add_argument("--epochs", default=os.getenv("EPOCHS","5"))
    return p.parse_args()

def wait_training(sm, name):
    while True:
        d = sm.describe_training_job(TrainingJobName=name)
        s = d["TrainingJobStatus"]
        print(f"[wait] training status={s}")
        if s in ("Completed","Failed","Stopped"):
            return d
        time.sleep(30)

def ensure_endpoint(sm, endpoint_name, cfg_name):
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        print(f"Updating endpoint: {endpoint_name} -> {cfg_name}")
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=cfg_name)
    except sm.exceptions.ClientError:
        print(f"Creating endpoint: {endpoint_name} -> {cfg_name}")
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=cfg_name)

def verify_live_image(sm, endpoint_name, must_contain_repo_fragment):
    ep_cfg = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointConfigName"]
    mdl = sm.describe_endpoint_config(EndpointConfigName=ep_cfg)["ProductionVariants"][0]["ModelName"]
    cont = sm.describe_model(ModelName=mdl)["PrimaryContainer"]
    live_img = cont.get("Image", "")
    live_art = cont.get("ModelDataUrl", "")
    print(f"LIVE_IMAGE={live_img}")
    print(f"LIVE_MODEL_DATA={live_art}")
    if f"/{must_contain_repo_fragment}:" not in live_img:
        raise RuntimeError(f"Endpoint attached WRONG image: {live_img} (expected repo fragment '{must_contain_repo_fragment}')")

def main():
    a = parse_args()
    region = a.region
    sm = boto3.client("sagemaker", region_name=region)

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    job_name = f"urbansound-train-{ts}"
    out_path = f"s3://{a.bucket}/artifacts/{job_name}"

    print(f"TRAIN_IMAGE: {a.train_image_uri}")
    print(f"INFER_IMAGE: {a.infer_image_uri}")

    # ---- TRAIN ----
    input_cfg = [{
        "ChannelName": "training",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri":      f"s3://{a.bucket}/training/",
                "S3DataDistributionType": "FullyReplicated",
            }
        },
        "InputMode": "File"
    }]

    print(f"Starting training job: {job_name}")
    sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=a.execution_role,
        AlgorithmSpecification={
            "TrainingImage": a.train_image_uri,
            "TrainingInputMode": "File",
        },
        InputDataConfig=input_cfg,
        OutputDataConfig={"S3OutputPath": out_path},
        ResourceConfig={
            "InstanceType": a.train_instance,
            "InstanceCount": 1,
            "VolumeSizeInGB": 50,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 2*3600},
        HyperParameters={"EPOCHS": str(a.epochs)},
        Environment={
            "S3_TRAIN_BUCKET": a.bucket,
            "S3_TRAIN_PREFIX": "training/",
            "SM_MODEL_DIR": "/opt/ml/model",
        },
        Tags=[
            {"Key":"project","Value":"urbansound"},
            {"Key":"commit","Value":a.git_sha},
            {"Key":"env","Value":a.env},
        ],
    )

    desc = wait_training(sm, job_name)
    if desc["TrainingJobStatus"] != "Completed":
        raise RuntimeError(f"Training failed: {desc.get('FailureReason')}")
    artifact = desc["ModelArtifacts"]["S3ModelArtifacts"]
    print(f"Model artifact: {artifact}")

    # ---- MODEL (INFERENCE IMAGE) ----
    model_name = f"{a.endpoint_base}-mdl-{ts}"
    print(f"Creating model: {model_name} (INFER image)")
    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=a.execution_role,
        PrimaryContainer={
            "Image": a.infer_image_uri,
            "Mode": "SingleModel",
            "ModelDataUrl": artifact,
        },
    )

    # ---- ENDPOINT CONFIG + ENDPOINT ----
    endpoint_name = f"{a.endpoint_base}-{a.env}"
    cfg_name = f"{endpoint_name}-cfg-{ts}"
    print(f"Creating endpoint-config: {cfg_name}")
    sm.create_endpoint_config(
        EndpointConfigName=cfg_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InstanceType": a.serve_instance,
            "InitialInstanceCount": 1,
            "InitialVariantWeight": 1.0,
            # critical so uvicorn+torch can start cleanly
            "ContainerStartupHealthCheckTimeoutInSeconds": 600,
        }],
    )

    ensure_endpoint(sm, endpoint_name, cfg_name)
    print("Waiting for endpoint InService...")
    sm.get_waiter("endpoint_in_service").wait(EndpointName=endpoint_name)

    # ---- VERIFY live image is inference repo ----
    verify_live_image(sm, endpoint_name, must_contain_repo_fragment="urbansound-infer")
    print(f"Endpoint live: {endpoint_name}")

if __name__ == "__main__":
    main()

