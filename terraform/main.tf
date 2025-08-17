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
data "aws_iam_policy_document" "sm_policy" {
  statement {
    sid       = "S3List"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn]
  }
  statement {
    sid       = "S3RW"
    effect    = "Allow"
    actions   = ["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:AbortMultipartUpload"]
    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }
  statement {
    sid     = "ECRPull"
    effect  = "Allow"
    actions = ["ecr:GetAuthorizationToken","ecr:BatchCheckLayerAvailability","ecr:GetDownloadUrlForLayer","ecr:BatchGetImage"]
    resources = ["*"]
  }
  statement {
    sid     = "Logs"
    effect  = "Allow"
    actions = ["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents","logs:DescribeLogStreams"]
    resources = ["arn:${data.aws_partition.p.partition}:logs:${local.region}:${local.account_id}:log-group:/aws/sagemaker/*"]
  }
}
resource "aws_iam_policy" "sm_inline" {
  name   = "${local.project}-sagemaker-exec-policy"
  policy = data.aws_iam_policy_document.sm_policy.json
}
resource "aws_iam_role_policy_attachment" "sm_attach" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = aws_iam_policy.sm_inline.arn
}

# GitHub OIDC provider with correct thumbprints
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1",
    "a031c46782e6e6c662c2c87c76da9aa62ccabd8e"
  ]
}

# GitHub Actions role with wildcard sub matching your exact repo name
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

# CI permissions
data "aws_iam_policy_document" "gha_policy" {
  statement {
    sid     = "ECRPushPull"
    effect  = "Allow"
    actions = [
      "ecr:GetAuthorizationToken","ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer","ecr:BatchGetImage","ecr:PutImage",
      "ecr:CompleteLayerUpload","ecr:InitiateLayerUpload","ecr:UploadLayerPart"
    ]
    resources = ["*"]
  }
  statement {
    sid     = "SageMakerOps"
    effect  = "Allow"
    actions = [
      "sagemaker:CreateTrainingJob","sagemaker:DescribeTrainingJob",
      "sagemaker:ListTrainingJobs","sagemaker:CreateModel","sagemaker:DeleteModel",
      "sagemaker:DescribeModel","sagemaker:CreateEndpointConfig",
      "sagemaker:DescribeEndpointConfig","sagemaker:CreateEndpoint",
      "sagemaker:UpdateEndpoint","sagemaker:DescribeEndpoint"
    ]
    resources = ["*"]
  }
  statement {
    sid     = "PassRole"
    effect  = "Allow"
    actions = ["iam:PassRole"]
    resources = [aws_iam_role.sagemaker_execution.arn]
  }
  statement {
    sid     = "S3List"
    effect  = "Allow"
    actions = ["s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn]
  }
  statement {
    sid     = "S3RW"
    effect  = "Allow"
    actions = ["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:AbortMultipartUpload"]
    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }
  statement {
    sid     = "Logs"
    effect  = "Allow"
    actions = ["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents","logs:DescribeLogStreams"]
    resources = ["arn:${data.aws_partition.p.partition}:logs:${local.region}:${local.account_id}:log-group:/aws/sagemaker/*"]
  }
}
resource "aws_iam_policy" "gha_inline" {
  name   = "${local.project}-gha-policy"
  policy = data.aws_iam_policy_document.gha_policy.json
}
resource "aws_iam_role_policy_attachment" "gha_attach" {
  role       = aws_iam_role.gha.name
  policy_arn = aws_iam_policy.gha_inline.arn
}

output "github_actions_role_arn" {
  value = aws_iam_role.gha.arn
}

