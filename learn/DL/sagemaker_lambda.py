import base64
import os

import boto3


def lambda_handler(event, context):
    aws_region = "us-east-1"
    notebook_name = event["notebook_name"]
    # s3_bucket='automate-fine-tunning-gblpoc'

    notebook_file = "lab-notebook.ipynb"
    iam = boto3.client("iam")

    # Create SageMaker and S3 clients
    sagemaker = boto3.client("sagemaker", region_name=aws_region)
    s3 = boto3.resource("s3", region_name=aws_region)
    s3_client = boto3.client("s3")
    s3_bucket = os.environ["s3_bucket"]
    s3_prefix = "notebook_lifecycle"

    lifecycle_config_script = f"""#!/bin/bash
set -e
sudo -u ec2-user -i <<EOF
cd /home/ec2-user/SageMaker/
aws s3 cp s3://{s3_bucket}/{s3_prefix}/training_scripts.zip .
unzip training_scripts.zip

echo "Create new env"
/home/ec2-user/anaconda3/condabin/conda create -n lab_310 python=3.10.14 -y 
source /home/ec2-user/anaconda3/bin/activate lab_310

conda install -y zeromq pyzmq

conda install -y pyarrow

pip3 install --upgrade pip==24.3.1
echo "Pip install ..."        
pip install -r requirements.txt
pip install nbconvert
echo "Running training job..."      
nohup jupyter nbconvert /home/ec2-user/SageMaker/lab-notebook.ipynb --to notebook --execute >> /home/ec2-user/SageMaker/nohup.out 2>&1 & 

conda deactivate

EOF
"""

    lifecycle_config_name = f"LCF-{notebook_name}"
    print(lifecycle_config_script)

    # Function to manage lifecycle configuration
    def manage_lifecycle_config(lifecycle_config_script):
        content = base64.b64encode(lifecycle_config_script.encode("utf-8")).decode(
            "utf-8"
        )
        try:
            # Create lifecycle configuration if not found
            sagemaker.create_notebook_instance_lifecycle_config(
                NotebookInstanceLifecycleConfigName=lifecycle_config_name,
                OnCreate=[{"Content": content}],
            )
        except sagemaker.exceptions.ClientError as e:
            print(e)

    # Try to describe the notebook instance to determine its status
    # Get the role with the specified name
    try:
        role = iam.get_role(RoleName="sagemaker_exec_role")
        sagemaker_exec_role = role["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        print("The role 'sagemaker_exec_role' does not exist.")

    try:
        response = sagemaker.describe_notebook_instance(
            NotebookInstanceName=notebook_name
        )
    except sagemaker.exceptions.ClientError as e:
        print(e)
        if "RecordNotFound" in str(e):
            manage_lifecycle_config(lifecycle_config_script)
            # Create a new SageMaker notebook instance if not found
            sagemaker.create_notebook_instance(
                NotebookInstanceName=notebook_name,
                InstanceType="ml.g5.4xlarge",
                RoleArn=sagemaker_exec_role,
                PlatformIdentifier="notebook-al2-v3",
                LifecycleConfigName=lifecycle_config_name,
                VolumeSizeInGB=30,
            )

        else:
            raise

    return {
        "statusCode": 200,
        "body": "Notebook instance setup and lifecycle configuration applied.",
    }