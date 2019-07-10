# Neural F0 model for synthetic Polish read speech
This projects aims at training a deep neural network for modelling Polish read speech F0.
Currently we are using AWS SageMaker as the cloud service for running both the training and
inference. 

## Build containers
Build and push a docker container to the Amazon Web Services ECR repository
`./build_and_push.sh`

## Run training on AWS SageMaker
