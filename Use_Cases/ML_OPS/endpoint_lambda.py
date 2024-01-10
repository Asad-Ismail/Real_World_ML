import json

region = sagemaker.Session().boto_region_name
boto_session = boto3.Session(region_name=region)
sagemaker_boto_client = boto_session.client("sagemaker")

def lambda_handler(event, context):
    
    model_package_arn = event['detail']['ModelPackageArn']
    response = sagemaker_boto_client.describe_model_package(ModelPackageName=model_package_arn)
    model_name = response['ModelPackageGroupName']
    model_data_url = response['InferenceSpecification']['Containers'][0]['ModelDataUrl']
    model_container= response['InferenceSpecification']['Containers'][0]['Image']
    
    # Create a model in SageMaker
    create_model_response = sagemaker_boto_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'Image': model_container,
            'ModelDataUrl': model_data_url
        },
        ExecutionRoleArn=sagemaker_iam_role
    )
    # Define endpoint configuration
    endpoint_config_name = model_name + 'deploy-config-1'
    sagemaker_boto_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'InstanceType': 'ml.m5.large', # Specify the instance type
                'InitialInstanceCount': 1,
                'ModelName': model_name,
                'VariantName': 'AllTraffic'
            }
        ]
    )
    
    # Create the endpoint
    endpoint_name = model_name + '-endpoint-1'
    sagemaker_boto_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
