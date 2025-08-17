# UrbanSound Audio Classification MLOps Pipeline

A complete MLOps pipeline for training and deploying audio classification models using AWS SageMaker, ECR, and GitHub Actions with OIDC authentication.

## 🚀 Overview

This project implements an end-to-end machine learning operations pipeline for audio classification using the UrbanSound8K dataset. The pipeline includes:

- **Containerized Training**: Custom Docker containers for model training on SageMaker
- **Containerized Inference**: Real-time inference endpoints 
- **CI/CD Pipeline**: GitHub Actions workflow with AWS OIDC authentication
- **Infrastructure as Code**: Terraform for AWS resource management
- **Model Deployment**: Automated SageMaker endpoint deployment

## 🏗️ Architecture

```
GitHub Actions (OIDC) → AWS ECR → SageMaker Training Job → Model Artifacts → SageMaker Endpoint
                    ↘ S3 Bucket ↗                      ↘ CloudWatch Logs
```

## 🛠️ Tech Stack

- **Cloud**: AWS (SageMaker, ECR, S3, IAM)
- **Containerization**: Docker
- **ML Framework**: PyTorch/TensorFlow
- **Audio Processing**: librosa, torchaudio
- **CI/CD**: GitHub Actions
- **IaC**: Terraform
- **Authentication**: GitHub OIDC → AWS IAM

## 📋 Prerequisites

- AWS Account with appropriate permissions
- GitHub repository with Actions enabled
- Terraform >= 1.5.0
- Docker
- Python 3.11+

## ⚙️ Setup

### 1. Infrastructure Deployment

```bash
# Clone the repository
git clone https://github.com/si3mshady/si3mshady-ai-in-the-sky-mlops-oidc.git
cd si3mshady-ai-in-the-sky-mlops-oidc

# Deploy AWS infrastructure
cd infra
terraform init
terraform plan
terraform apply
```

### 2. GitHub Repository Configuration

#### Enable Workflow Permissions
1. Go to **Settings → Actions → General**
2. Under **Workflow permissions**, select **"Read and write permissions"**
3. Click **Save**

#### Configure Repository Secrets (Optional)
- `ACTIONS_STEP_DEBUG=true` (for detailed logs)

### 3. Environment Variables

Update the workflow file `.github/workflows/mlops.yml` with your AWS account details:

```yaml
env:
  AWS_REGION: us-east-2
  AWS_ACCOUNT_ID: "your-account-id"
  S3_BUCKET: your-s3-bucket-name
  TRAIN_ECR_REPOSITORY: urbansound-train
  INFER_ECR_REPOSITORY: urbansound-infer
  SAGEMAKER_EXECUTION_ROLE_ARN: arn:aws:iam::your-account:role/urbansound-sagemaker-execution
  OIDC_ROLE_ARN: arn:aws:iam::your-account:role/urbansound-github-actions-v2
```

## 🚀 Usage

### Trigger Training Pipeline

#### Automatic Triggers
- **Push to main/develop**: Automatically triggers the full pipeline
- **Pull Request**: Runs validation without deployment

#### Manual Triggers
```bash
# Via GitHub UI: Actions → UrbanSound MLOps → Run workflow
# Or via GitHub CLI:
gh workflow run mlops.yml
```

#### Manual Trigger with Parameters
```bash
gh workflow run mlops.yml \
  -f force_retrain=true \
  -f endpoint_env=production \
  -f train_instance_type=ml.m5.xlarge \
  -f serve_instance_type=ml.g4dn.2xlarge
```

### Monitor Pipeline

1. **GitHub Actions**: View workflow progress in the Actions tab
2. **AWS Console**: 
   - SageMaker → Training jobs
   - SageMaker → Endpoints  
   - CloudWatch → Log groups (`/aws/sagemaker/*`)
   - ECR → Repositories

## 📁 Project Structure

```
.
├── .github/workflows/
│   └── mlops.yml                 # GitHub Actions pipeline
├── docker/
│   ├── training/
│   │   ├── Dockerfile           # Training container
│   │   ├── requirements.txt     # Training dependencies
│   │   └── entryPoint.py        # Training script
│   └── inference/
│       ├── Dockerfile           # Inference container  
│       ├── requirements.txt     # Inference dependencies
│       └── predictor.py         # Inference logic
├── infra/
│   └── main.tf                  # Terraform infrastructure
├── scripts/
│   └── sm_train_and_deploy.py   # SageMaker orchestration
└── README.md
```

## 🔧 Configuration

### Training Configuration

Modify training parameters in the workflow dispatch inputs or environment variables:

- `train_instance_type`: SageMaker training instance (default: `ml.m5.large`)  
- `serve_instance_type`: SageMaker inference instance (default: `ml.g4dn.xlarge`)
- `force_retrain`: Force retraining even if model exists (default: `false`)
- `endpoint_env`: Deployment environment (`staging`/`production`)

### Model Configuration

Update training parameters in `docker/training/entryPoint.py`:

```python
# Model hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 10  # UrbanSound8K classes
```

## 🔐 Security Features

- **OIDC Authentication**: No long-lived AWS credentials in GitHub
- **IAM Role-Based Access**: Minimal permissions following least privilege
- **Temporary Credentials**: Short-lived tokens for AWS access  
- **Private ECR**: Container images stored in private repositories

## 🐛 Troubleshooting

### Common Issues

#### OIDC Authentication Fails
```bash
# Verify IAM trust policy allows your repository
aws iam get-role --role-name urbansound-github-actions-v2
```

#### Docker Build Fails
```bash
# Check file paths in Dockerfile match your structure  
# Ensure context is set correctly in workflow
context: docker/training  # or adjust COPY paths
```

#### SageMaker Training Fails
```bash
# Check CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix "/aws/sagemaker"
```

### Debug Mode

Enable detailed logging:
```bash
# Add repository secret
ACTIONS_STEP_DEBUG=true
```

## 📊 Monitoring & Metrics

- **Training Metrics**: Logged to CloudWatch during training
- **Endpoint Metrics**: Available in SageMaker console
- **Cost Monitoring**: Use AWS Cost Explorer to track expenses
- **Performance**: Monitor inference latency and throughput

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/)
- [AWS SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

**Built with ❤️ for MLOps and Audio AI**
