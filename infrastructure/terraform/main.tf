terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Deep Learning Base AMI with Single CUDA (Ubuntu 22.04)
# AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
locals {
  dlami_id = "ami-0c9c917c544c180e8"
}

# Security group for EC2 instance (SSM - no inbound ports needed)
resource "aws_security_group" "ti2v_sg" {
  name        = "ti2v-inference-sg"
  description = "Security group for TI2V inference instance (SSM access only)"

  # No ingress rules - SSM Session Manager uses outbound HTTPS only

  egress {
    description = "Allow all outbound traffic (required for SSM)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "ti2v-inference-sg"
  }
}

# IAM role for EC2 to use SSM
resource "aws_iam_role" "ssm_role" {
  name = "ti2v-ssm-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })

  tags = {
    Name = "ti2v-ssm-role"
  }
}

# Attach AWS managed policy for SSM
resource "aws_iam_role_policy_attachment" "ssm_policy" {
  role       = aws_iam_role.ssm_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# S3 policy for uploading/downloading video outputs
resource "aws_iam_role_policy" "s3_access" {
  name = "ti2v-s3-access"
  role = aws_iam_role.ssm_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:DeleteObject"
        ]
        Resource = "arn:aws:s3:::${var.s3_bucket_name}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = "arn:aws:s3:::${var.s3_bucket_name}"
      }
    ]
  })
}

# Instance profile for EC2
resource "aws_iam_instance_profile" "ssm_profile" {
  name = "ti2v-ssm-profile"
  role = aws_iam_role.ssm_role.name

  tags = {
    Name = "ti2v-ssm-profile"
  }
}

# EC2 instance
resource "aws_instance" "ti2v_inference" {
  ami           = local.dlami_id
  instance_type = var.instance_type

  availability_zone = var.availability_zone

  vpc_security_group_ids = [aws_security_group.ti2v_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.ssm_profile.name

  root_block_device {
    volume_size = 500
    volume_type = "gp3"
  }

  user_data = file("${path.module}/user_data.sh")

  tags = {
    Name = "ti2v-inference"
  }
}
