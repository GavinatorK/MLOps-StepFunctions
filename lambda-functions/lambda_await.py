import json

import boto3
import os

sagemaker = boto3.client('sagemaker')

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



def lambda_handler(event, context):
    stage = event['stage']
    if stage == 'Training':
        name = event['training_job_name']
        training_details = describe_training_job(name)
        print(training_details)
        status = training_details['TrainingJobStatus']
        if status == 'Completed':
            model_data_url = training_details['ModelArtifacts']['S3ModelArtifacts']
            event['message'] = 'Training job "{}" complete. Model data uploaded to "{}"'.format(name, model_data_url)
            event['model_data_url'] = model_data_url
        elif status == 'Failed':
            failure_reason = training_details['FailureReason']
            event['message'] = 'Training job failed. {}'.format(failure_reason)
    elif stage == 'BatchTransform':
        name=event['transform_job_name']
        transform_details=describe_transform_job(name)
        status= transform_details['TransformJobStatus']
        if status=="Completed":
            event['message']= 'Batch Transfrom completed {}'.format(name)
            event['transform_output']=transform_details['TransformOutput']['S3OutputPath']
        elif status=="Failed":
            failure_reason=transform_details['FailureReason']
            event['message']='Transform Job Failed. {}'.format(failure_reason)
    elif stage == 'Deployment':
        name = event['endpoint']
        endpoint_details = describe_endpoint(name)
        status = endpoint_details['EndpointStatus']
        if status == 'InService':
            event['message'] = 'Deployment completed for endpoint "{}".'.format(name)
        elif status == 'Failed':
            failure_reason = endpoint_details['FailureReason']
            event['message'] = 'Deployment failed for endpoint "{}". {}'.format(name, failure_reason)
        elif status == 'RollingBack':
            event['message'] = 'Deployment failed for endpoint "{}", rolling back to previously deployed version.'.format(name)
    elif stage=="Processing":
        name=event['processing_job_name']
        processing_details=describe_processing(name)
        status=processing_details['ProcessingJobStatus']
        if status=="Completed":
            event['processing_output']=processing_details['ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri']
            event['proc_job_status']="Success"
        elif status=="Failed":
            failure_reason=processing_details['FailureReason']
            event['message']='Processing Job Failed. {}'.format(failure_reason)
            event['proc_job_status']="Failed"
        update_ddb_item(event['name'], event['date-time'],'processing_info', event['proc_job_status'])
            
    event['status'] = status
    return event




def describe_training_job(name):
    """ Describe SageMaker training job identified by input name.
    Args:
        name (string): Name of SageMaker training job to describe.
    Returns:
        (dict)
        Dictionary containing metadata and details about the status of the training job.
    """
    try:
        response = sagemaker.describe_training_job(
            TrainingJobName=name
        )
    except Exception as e:
        print(e)
        print('Unable to describe hyperparameter tunning job.')
        raise(e)
    return response
    
    
def describe_transform_job(name):
    """ Describe SageMaker transform job identified by input name.
    Args:
        name (string): Name of SageMaker endpoint to describe.
    Returns:
        (dict)
        Dictionary containing metadata and details about the status of the transform job.
    """
    
    try:
        response = sagemaker.describe_transform_job(
            TransformJobName=name
        )
    except Exception as e:
        print(e)
        print('Unable to describe endpoint.')
        raise(e)
    return response
    
    
    

def describe_endpoint(name):
    """ Describe SageMaker endpoint identified by input name.
    Args:
        name (string): Name of SageMaker endpoint to describe.
    Returns:
        (dict)
        Dictionary containing metadata and details about the status of the endpoint.
    """
    try:
        response = sagemaker.describe_endpoint(
            EndpointName=name
        )
    except Exception as e:
        print(e)
        print('Unable to describe endpoint.')
        raise(e)
    return response
    

def describe_processing(name):
    """ Describe SageMaker processing identified by input name.
    Args:
        name (string): Name of SageMaker processing to describe.
    Returns:
        (dict)
        Dictionary containing metadata and details about the status of the endpoint.
    """
    try:
        response = sagemaker.describe_processing_job(
            
            ProcessingJobName=name
        )
    except Exception as e:
        print(e)
        print('Unable to describe processing.')
        raise(e)
    return response
    

def update_ddb_item(modelName,dateTime,update_field, update_val, new_field=None):
    response = table.update_item(
    Key={
        'modelName': modelName,
        'dateTime': dateTime
    },
    UpdateExpression=f"set {update_field}.progress=:p",
    ExpressionAttributeValues={
        ':p': update_val
    },
    ReturnValues="ALL_NEW"
    )
    
    print("UpdateItem succeeded:")
    print(json.dumps(response, indent=4, cls=DecimalEncoder))
    
    