## Applying Voice Classification in Amazon Connect Contact Flows

## Overview Architecture
![Architecture](images/connect_audio_stream_ml_inference.jpg)

## Deployment Steps

### Step 1: Create SageMaker Notebook Instance to Train and Deploy Voice Classification Model

You can create the SageMaker Notebook instance using 1-click deployment button:
[![launchstackbutton](images/launchbutton.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/template?stackName=SageMakerNotebookInstanceStack&templateURL=https://connect-voice-classification.s3.amazonaws.com/sagemaker_template.yaml
)

### Step 2: Create the Lambda Functions to Process Live Media Streaming and Make Inference
[![launchstackbutton](images/launchbutton.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/template?stackName=ConnectLiveAudioStreamInferenceStack&templateURL=https://connect-voice-classification.s3.amazonaws.com/live_audio_streaming_cfn.yaml
)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

