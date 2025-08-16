########################################
# infra/main.tf  (single file)
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

# --------- EDIT THESE 3 VALUES ----------
locals {
  s3_bucket_name = "urbansound-mlops-56423506"     # <--- change if taken
  github_owner   = "si3mshady"                     # <--- e.g., johndoe
  github_repo    = "ai-in-the-sky-mlops-oidc"             # <--- your repo name
  project        = "urbansound"
  region         = "us-east-2"
  account_id     = "564230509626"
}

# ---------------------------------------
data "aws_partition" "p" {}
data "aws_caller_identity" "me" {}

# S3 artifacts bucket
resource "aws_s3_bucket" "artifacts" { 
  bucket = local.s3_bucket_name 
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { 
    status = "Enabled" 
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ECR repos (train & infer)
resource "aws_ecr_repository" "train" {
  name = "${local.project}-train"
  image_scanning_configuration { 
    scan_on_push = true 
  }
}

resource "aws_ecr_repository" "infer" {
  name = "${local.project}-infer"
  image_scanning_configuration { 
    scan_on_push = true 
  }
}

resource "aws_ecr_lifecycle_policy" "train" {
  repository = aws_ecr_repository.train.name
  policy = jsonencode({ 
    rules = [{
      rulePriority = 1
      description = "Keep last 20"
      selection = {
        tagStatus = "any"
        countType = "imageCountMoreThan"
        countNumber = 20
      }
      action = {
        type = "expire"
      }
    }]
  })
}

resource "aws_ecr_lifecycle_policy" "infer" {
  repository = aws_ecr_repository.infer.name
  policy = jsonencode({ 
    rules = [{
      rulePriority = 1
      description = "Keep last 20"
      selection = {
        tagStatus = "any"
        countType = "imageCountMoreThan"
        countNumber = 20
      }
      action = {
        type = "expire"
      }
    }]
  })
}

# SageMaker Execution Role (used by training & endpoint containers)
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
    actions   = ["s3:ListBucket"]
    effect    = "Allow"
    resources = [aws_s3_bucket.artifacts.arn]
  }
  
  statement {
    sid       = "S3RW"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"]
    effect    = "Allow"
    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }
  
  statement {
    sid       = "ECRPull"
    actions   = ["ecr:GetAuthorizationToken", "ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"]
    effect    = "Allow"
    resources = ["*"]
  }
  
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    effect    = "Allow"
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

# GitHub OIDC provider (if not already present)
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

# GitHub Actions role (assumed by workflows via OIDC)
data "aws_iam_policy_document" "gha_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${local.github_owner}/${local.github_repo}:*"]
    }
  }
}

resource "aws_iam_role" "gha" {
  name               = "${local.project}-github-actions"
  assume_role_policy = data.aws_iam_policy_document.gha_assume.json
}

data "aws_iam_policy_document" "gha_policy" {
  statement {
    sid    = "ECRPushPull"
    effect = "Allow"
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:PutImage",
      "ecr:CompleteLayerUpload",
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart"
    ]
    resources = ["*"]
  }
  
  statement {
    sid    = "SMOps"
    effect = "Allow"
    actions = [
      "sagemaker:CreateTrainingJob",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:ListTrainingJobs",
      "sagemaker:CreateModel",
      "sagemaker:DeleteModel",
      "sagemaker:DescribeModel",
      "sagemaker:CreateEndpointConfig",
      "sagemaker:DeleteEndpointConfig",
      "sagemaker:DescribeEndpointConfig",
      "sagemaker:CreateEndpoint",
      "sagemaker:UpdateEndpoint",
      "sagemaker:DeleteEndpoint",
      "sagemaker:DescribeEndpoint",
      "sagemaker:ListEndpoints"
    ]
    resources = ["*"]
  }
  
  statement {
    sid       = "PassRole"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.sagemaker_execution.arn]
  }
  
  statement {
    sid       = "S3List"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn]
  }
  
  statement {
    sid       = "S3RW"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"]
    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }
  
  statement {
    sid    = "Logs"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams"
    ]
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

# Outputs to paste into GitHub workflow
output "s3_bucket_name" { 
  value = aws_s3_bucket.artifacts.bucket 
}

output "ecr_train_repository_url" { 
  value = aws_ecr_repository.train.repository_url 
}

output "ecr_infer_repository_url" { 
  value = aws_ecr_repository.infer.repository_url 
}

output "sagemaker_execution_role_arn" { 
  value = aws_iam_role.sagemaker_execution.arn 
}

output "github_actions_role_arn" { 
  value = aws_iam_role.gha.arn 
}

