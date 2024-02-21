## Infrastucture as Code

Infrastructure as Code (IaC) is the practice of managing infrastructure using code-based configuration files rather than manual processes. With IaC, infrastructure resources are defined, provisioned, and managed through code, enabling automation, version control, and consistency across environments.

Two of most common IaCplaforms are 


1. Terraform
2. Cloudformation

### Terraform

Terraform is a prominent IaC tool that facilitates this approach by allowing users to define infrastructure configurations in declarative configuration files, which Terraform then translates into API calls to provision and manage the desired infrastructure resources.

**Main components of Terraform**

**Provider Block**:
 Specifies the cloud or service provider that Terraform will use to provision resources.

  ```
provider "aws" {
  region = "us-west-2"
}

 ```

**Resource Blocks**:
 Define the desired infrastructure resources (e.g., virtual machines, networks, databases) and their configurations.

  ```
 resource "aws_instance" "example" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
}
  ```

**Variable Declarations**:
 Define input variables used to parameterize the configuration and make it reusable.

  ```
 variable "instance_count" {
  description = "The number of EC2 instances to create"
  type        = number
  default     = 2
}

  ```


**Output Values**: Define output values that can be extracted from the infrastructure after it's provisioned.

  ```
output "instance_public_ip" {
  value = aws_instance.example.public_ip
}
  ```

**Modules**: Encapsulate reusable components of Terraform configurations for better organization and abstraction.
  
  ```
modules/
├── ec2_instance
│   ├── main.tf
│   └── variables.tf
└── vpc
    ├── main.tf
    └── variables.tf
  ```

**TerraForm Workflow**

 ```
# Initialize Terraform in the working directory
terraform init

# Generate and preview the execution plan
terraform plan

# Apply the changes to provision the infrastructure
terraform apply

# Destroy the provisioned infrastructure
terraform destroy
 ```