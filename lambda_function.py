
import json
import boto3


runtime = boto3.Session().client('sagemaker-runtime')
endpoint_name = 'mnist-endpoint'

def lambda_handler(event, context):
    print('Context:::', context)
    print('EventType::', type(event))
    
    response = runtime.invoke_endpoint(
        EndpointName = endpoint_name,
        ContentType = 'application/json',
        Accept = 'application/json',
        Body = json.dumps(event)
    )
    
    result = response['Body'].read().decode('utf-8')
    result = json.loads(result)
    
    return {
        'statusCode': 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'type-result':str(type(result)),
        'Content-Type-In':str(context),
        'body' : json.dumps(result),
        }
