import os
import json
import time
import boto3
import httpx
import requests
import configparser
from tinydb import TinyDB, Query
from urllib.parse import unquote_plus

# Initialize TinyDB
db = TinyDB('session_histories.json')
# Define table for file paths
s3_file_path = db.table('s3_file_path')

def sync_s3_to_local(bucket_name, object_key, event_name, message):
    local_directory, s3_folder_prefix = None, None
    # Check if the object_key starts with the s3_folder_prefix
    for record in s3_file_path:
        # Check if the variable_value starts with the s3_path value
        if object_key.startswith(record['s3_path']):
            # Perform actions if a match is found
            local_directory = record['local_path']
            s3_folder_prefix = record['s3_path']
            client_name = record['client_name']
    
    local_file_path = os.path.join(local_directory, object_key[len(s3_folder_prefix):])

    try:
        if event_name.startswith('ObjectCreated'):
            # Download the file from S3
            s3_client.download_file(bucket_name, object_key, local_file_path)
            print(f"Downloaded {object_key} to {local_file_path}")
            data = {
                "file_path": local_file_path,
                "file_name": object_key[len(s3_folder_prefix):]
            }

            url = api_url + f"/update_vector_in_chromadb/?collection_name={client_name}"
            response = requests.post(url, json=data)
            if response.status_code == 200:
                print(f"File {local_file_path} vectorized successfully.")
            else:
                print(f"Failed to vectorize file {local_file_path}.")
        elif event_name.startswith('ObjectRemoved'):
            # Delete the file from the local directory
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
                print(f"Deleted {local_file_path}")
            filename = object_key[len(s3_folder_prefix):],
            collection_name = client_name
            url = api_url + f"/delete_vector_in_chromadb/?filename={filename}&collection_name={collection_name}"
            response = requests.delete(url)
            if response.status_code == 200:
                print(f"Vector for file {local_file_path} removed successfully.")
            else:
                print(f"Failed to remove file {local_file_path}.")
    except s3_client.exceptions.NoSuchKey:
        print(f"The object {object_key} does not exist in S3.")
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"The object {object_key} was not found in S3.")
        else:
            raise  # Re-raise the exception if it's not a '404' error
    finally:
        # Delete the message from the queue
        try:
            print(message['ReceiptHandle'])
            sqs_client.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=message['ReceiptHandle']
            )
            print(f"Deleting {local_file_path} from SQS queue")
        except KeyError:
            print("Message not found in SQS queue")

def poll_sqs_queue():
    while True:
        # Receive messages from SQS queue
        response = sqs_client.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
            MessageAttributeNames=['All']
        )
        
        if 'Messages' in response:
            for message in response['Messages']:
                try:
                    # Parse the message body
                    message_body = json.loads(message['Body'])
                    event_name = message_body['Records'][0]['eventName']
                except s3_client.exceptions.NoSuchKey:
                    print(f"The object {object_key} does not exist in S3.")
                except s3_client.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == '404':
                        print(f"The object {object_key} was not found in S3.")
                    else:
                        raise # Re-raise the exception if it's not a '404' error
                
                # Check if the event is ObjectCreated or ObjectRemoved
                if not event_name.startswith(('ObjectCreated', 'ObjectRemoved')):
                    print(f"Ignoring event {event_name}")
                    continue
                
                bucket_name = message_body['Records'][0]['s3']['bucket']['name']
                object_key = message_body['Records'][0]['s3']['object']['key']
                object_key = unquote_plus(object_key)
                
                # Sync S3 to local directory
                sync_s3_to_local(bucket_name, object_key, event_name, message)

        time.sleep(1)

if __name__ == '__main__':
    
    # Define the path to your config file
    config = configparser.ConfigParser()

    # Read the environment variables for the client from the toml file
    try:
        config.read("./config.ini")
        session = boto3.Session(profile_name="chatbot_aws")
        s3_client = session.client('s3')
        sqs_client = session.client('sqs')
        queue_url = config.get('DEFAULT', 'QUEUE_URL')
        api_url = config.get('DEFAULT', 'API_URL')
    except KeyError:
        print("Missing required environment variables")
        exit(1)
    
    poll_sqs_queue()
