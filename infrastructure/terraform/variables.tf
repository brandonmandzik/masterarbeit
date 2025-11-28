variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "availability_zone" {
  description = "AWS availability zone (must match region)"
  type        = string
  default     = "us-east-1a"
}

variable "instance_type" {
  description = "EC2 instance type (g6e.4xlarge or p5.4xlarge recommended)"
  type        = string
  default     = "g6e.4xlarge"
}

variable "s3_bucket_name" {
  description = "Name of existing S3 bucket for storing video outputs"
  type        = string
  default = "my-research-results"
}
