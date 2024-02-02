import sagemaker
import boto3

iam_client = boto3.client('iam')
role = iam_client.get_role(RoleName='run_biogpt')['Role']['Arn']
sess = sagemaker.Session()
