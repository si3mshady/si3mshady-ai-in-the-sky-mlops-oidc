########################################
# infra/main.tf
########################################

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.40"
    }
  }
}

provider "aws" {
  region = "us-east-2"
}

locals {
  project      = "urbansound"
  region       = "us-east-2"
  account_id   = "564230509626"
  s3_bucket    = "urbansound-mlops-56423506"
  github_owner = "si3mshady"
  github_repo  = "si3mshady-ai-in-the-sky-mlops-oidc"
}

data "aws_partition" "p" {}
data "aws_caller_identity" "me" {}

# S3 artifacts bucket
resource "aws_s3_bucket" "artifacts" {
  bucket = local.s3_bucket
}
resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}
resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}
resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ECR repositories
resource "aws_ecr_repository" "train" {
  name = "${local.project}-train"
  image_scanning_configuration { scan_on_push = true }
}
resource "aws_ecr_repository" "infer" {
  name = "${local.project}-infer"
  image_scanning_configuration { scan_on_push = true }
}

# SageMaker execution role
data "aws_iam_policy_document" "sm_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}
resource "aws_iam_role" "sagemaker_execution" {
  name               = "${local.project}-sagemaker-execution"
  assume_role_policy = data.aws_iam_policy_document.sm_assume.json
}
resource "aws_iam_role_policy_attachment" "sm_admin" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}
resource "aws_iam_role_policy_attachment" "sm_s3" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

# GitHub OIDC provider with correct thumbprints
resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1",
    "a031c46782e6e6c662c2c87c76da9aa62ccabd8e"
  ]
}

# GitHub Actions role with ADMIN PERMISSIONS
resource "aws_iam_role" "gha" {
  name = "${local.project}-github-actions-v2"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = aws_iam_openid_connect_provider.github.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
        StringLike = {
          "token.actions.githubusercontent.com:sub" = "repo:${local.github_owner}/${local.github_repo}:*"
        }
      }
    }]
  })
}

# FULL ADMIN ACCESS FOR GITHUB ACTIONS
resource "aws_iam_role_policy_attachment" "gha_admin" {
  role       = aws_iam_role.gha.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}

output "github_actions_role_arn" {
  value = aws_iam_role.gha.arn
}
