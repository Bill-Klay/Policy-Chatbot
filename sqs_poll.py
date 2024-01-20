import boto3
import json
import toml
import os

def sync_s3_to_local(bucket_name, object_key, event_name):
    # Check if the object_key starts with the s3_folder_prefix
    if not object_key.startswith(s3_folder_prefix):
        print(f"Ignoring {object_key} as it does not match the prefix {s3_folder_prefix}")
        return

    local_file_path = os.path.join(local_directory, object_key[len(s3_folder_prefix):])

    try:
        if event_name.startswith('ObjectCreated'):
            # Download the file from S3
            s3_client.download_file(bucket_name, object_key, local_file_path)
            print(f"Downloaded {object_key} to {local_file_path}")
        elif event_name.startswith('ObjectRemoved'):
            # Delete the file from the local directory
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
                print(f"Deleted {local_file_path}")
    except s3_client.exceptions.NoSuchKey:
        print(f"The object {object_key} does not exist in S3.")
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"The object {object_key} was not found in S3.")
        else:
            raise  # Re-raise the exception if it's not a '404' error

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
                # Parse the message body
                message_body = json.loads(message['Body'])
                event_name = message_body['Records'][0]['eventName']
                
                # Check if the event is ObjectCreated or ObjectRemoved
                if not event_name.startswith(('ObjectCreated', 'ObjectRemoved')):
                    print(f"Ignoring event {event_name}")
                    continue
                
                bucket_name = message_body['Records'][0]['s3']['bucket']['name']
                object_key = message_body['Records'][0]['s3']['object']['key']
                
                # Sync S3 to local directory
                sync_s3_to_local(bucket_name, object_key, event_name)
                
                # Delete the message from the queue
                sqs_client.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=message['ReceiptHandle']
                )

if __name__ == '__main__':
    
    # Define the path to your config file
    config_path = './config.toml'

    # Read the environment variables for the client from the toml file
    try:
        config = toml.load(config_path)
        session = boto3.Session(profile_name="chatbot_aws")
        s3_client = session.client('s3')
        sqs_client = session.client('sqs')
        queue_url = config['DEFAULT']['QUEUE_URL']
        local_directory = config['DEFAULT']['LOCAL_DIRECTORY']
        s3_folder_prefix = config['DEFAULT']['S3_FOLDER_PREFIX']  # Include the trailing slash in the toml
    except KeyError:
        print("Missing required environment variables")
        exit(1)
    
    poll_sqs_queue()
