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

**Resource Blocks**:
 Define the desired infrastructure resources (e.g., virtual machines, networks, databases) and their configurations.

**Variable Declarations**:
 Define input variables used to parameterize the configuration and make it reusable.

**Output Values**: Define output values that can be extracted from the infrastructure after it's provisioned.

**Modules**: Encapsulate reusable components of Terraform configurations for better organization and abstraction.