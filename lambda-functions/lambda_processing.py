import os
import json
import boto3
import time

from typing import Optional
import decimal


# Helper class to convert a DynamoDB item to JSON.
class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            if abs(o) % 1 > 0:
                return float(o)
            else:
                return int(o)
        return super(DecimalEncoder, self).default(o)

dynamodb = boto3.resource('dynamodb')

table = dynamodb.Table('plops-dev')




region = boto3.Session().region_name    
smclient = boto3.Session().client('sagemaker')
roleArn = "arn:aws:iam::xxxxxxxxx:role/mlops_stepfuncs_sm_s3"

proc_container='xxxxxxxxx.dkr.ecr.us-east-2.amazonaws.com/sagemaker-processing-container:latest'
#pass None if not providing
script_uri="s3://<your-bucket-name>/code/preprocessing.py"

bucket_path="s3://<your-bucket-name>"
prefix = "train"
input_s3_uri=bucket_path+"/"+prefix
output_s3_uri=bucket_path+"/processing_output"

#make sure this is unique for each execution
name="<base name for model>"

def get_unique_job_name(base_name: str):
    """ Returns a unique job name based on a given base_name
        and the current timestamp """
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    return f'{base_name}', f'{timestamp}'


def get_file_input(name: str, input_s3_uri: str, output_path: str):
    """ Returns the input file configuration
        Modify if you need different input method """
    return {
        'InputName': name,
        'S3Input': {
            'S3Uri': input_s3_uri,
            'LocalPath': output_path,
            'S3DataType': 'S3Prefix',
            'S3InputMode': 'File'
        }
    }

def get_file_output(name: str, local_path: str, ouput_s3_uri: str):
    """ Returns output file configuration
        Modify for different output method """
    return {
        'OutputName': name,
        'S3Output': {
            'S3Uri': ouput_s3_uri,
            'LocalPath': local_path,
            'S3UploadMode': 'EndOfJob'
        }
    }


def get_app_spec(image_uri: str, container_arguments: Optional[str], entrypoint: Optional[str]):
    app_spec = {
        'ImageUri': image_uri
    }
    
    if container_arguments is not None:
        app_spec['ContainerArguments'] = container_arguments

    if entrypoint is not None:
        # Similar to ScriptProcessor in sagemaker SDK:
        # Run a custome script within the container
        app_spec['ContainerEntrypoint'] = ['python3', entrypoint]

    return app_spec


def lambda_handler(event, context):

    # (1) Get inputs
    event['S3Input']=input_s3_uri
    event['S3Output'] = output_s3_uri
    event['ProcessingImageUri']=proc_container
    event['name']=name
    if script_uri is not None:
        event['script_uri']=script_uri  # Optional: S3 path to custom script

    # Get execution environment
    role = roleArn
    instance_type ="ml.m4.xlarge"
    volume_size = 20
    max_runtime = 3600  # Default: 1h
    container_arguments = ["--img_size", "1700", "--src_prefix", "Priority", "--dest_prefix", "abc"]
    entrypoint = None  # Entrypoint to the container, will be set automatically later
    
    
    
    job_name, timestamp = get_unique_job_name(name)  # (2)
    event['date_time']=timestamp
    job_name=job_name+"-"+timestamp


    #
    # (3) Specify inputs / Outputs
    #

    inputs = [
        get_file_input('data', input_s3_uri, '/opt/ml/processing/input')
    ]

    if script_uri is not None:
        # Add custome script to the container (similar to ScriptProcessor)
        inputs.append(get_file_input('script', script_uri, '/opt/ml/processing/code'))

        # Make script new entrypoint for the container
        filename = os.path.basename(script_uri)
        entrypoint = f'/opt/ml/processing/code/{filename}'

    outputs = [
        get_file_output('output_data', '/opt/ml/processing/output', output_s3_uri)
    ]

    #
    # Define execution environment
    #

    app_spec = get_app_spec(proc_container, container_arguments, entrypoint)

    cluster_config = {
        'InstanceCount': 1,
        'InstanceType': instance_type,
        'VolumeSizeInGB': volume_size
    }

    #
    # (4) Create processing job and return job ARN
    #
    smclient.create_processing_job(
        ProcessingInputs=inputs,
        ProcessingOutputConfig={
            'Outputs': outputs
        },
        ProcessingJobName=job_name,
        ProcessingResources={
            'ClusterConfig': cluster_config
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': max_runtime
        },
        AppSpecification=app_spec,
        RoleArn=role
    )
    
    event["stage"] = "Processing"
    event["status"] = "InProgress"
    event['name'] = name
    event['processing_job_name']=job_name
    
    
    response = table.put_item(
    Item={
        'modelName': name,
        'dateTime': timestamp,
        'processing_info': {
            'job_name':job_name,
            'container': proc_container,
            'input':input_s3_uri,
            'output':output_s3_uri
                }
         }
    )

    print("PutItem succeeded:")
    print(json.dumps(response, indent=4, cls=DecimalEncoder))
    return event
