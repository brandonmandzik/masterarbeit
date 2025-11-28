output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.ti2v_inference.id
}

output "instance_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.ti2v_inference.public_ip
}

output "instance_public_dns" {
  description = "Public DNS name of the EC2 instance"
  value       = aws_instance.ti2v_inference.public_dns
}

output "ssm_connection_command" {
  description = "AWS Systems Manager command to connect to the instance"
  value       = "aws ssm start-session --target ${aws_instance.ti2v_inference.id}"
}

output "s3_bucket_name" {
  description = "S3 bucket name for video outputs"
  value       = var.s3_bucket_name
}
