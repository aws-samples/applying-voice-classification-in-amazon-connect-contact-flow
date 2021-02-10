## AWS SAM Deployment for Lambda Functions

### Prerequisite 

Fulfill the following requirements:

- Install and Configure [AWS CLI](https://aws.amazon.com/cli/)
- Install AWS [SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html)
- Install the build tools: [Gradle](https://docs.gradle.org/current/userguide/installation.html) and [Node/NPM](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)

If you use [AWS Cloud9 IDE](https://aws.amazon.com/cloud9/), you just need to install Gradle. You can use [sdkman](https://sdkman.io/install):

`curl -s "https://get.sdkman.io" | bash`  
`source "$HOME/.sdkman/bin/sdkman-init.sh"`   
`sdk install gradle`

### Deployment 
`sam build` followed by `sam deploy --guided`