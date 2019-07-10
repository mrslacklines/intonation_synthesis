# Neural F0 model for synthetic Polish read speech
This projects aims at training a deep neural network for modelling Polish read speech F0.
Currently we are using AWS SageMaker as the cloud service for running both the training and
inference. 

## Build containers
Build and push a docker container to the Amazon Web Services ECR repository
`./build_and_push.sh`

or 

`docker build -f Dockerfile-dev . -t intonation-dev`
to build for local development.

## Run training locally
In order to run the training container locally you need to mount you dataset to the docker container to emulate the SageMaker environment, i.e.

If your data lives in `/media/tomaszk/DANE2/Speech_archive/HTS-demo_AMU_PL_ILO_STRAIGHT` on the host machine like mine does, you can simply:

`docker run -it -v /media/tomaszk/DANE2/Speech_archive/HTS-demo_AMU_PL_ILO_STRAIGHT:/opt/ml/input/training intonation-dev`

## Run training on AWS SageMaker

Open up your Notebook instance and run:
```
import boto3
import sagemaker as sage
from sagemaker import get_execution_role

role = get_execution_role()
sess = sage.Session()

account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name # or setup the region by hand
image = '{}.dkr.ecr.{}.amazonaws.com/intonation-staging'.format(account, region)

estimator = sage.estimator.Estimator(
    image,
    role, 1,
    'ml.m4.xlarge',
    train_volume_size=50,
    output_path="s3://{}/output".format(sess.default_bucket()),
    sagemaker_session=sess,
    hyperparameters={
    "epochs": 2,
    "learning_rate": 0.01,
    "num_layers": 1,
    "hidden_size": 64,
    "dropout": 0.2,
    "batch_size": 2,
    "bidirectional": False
    })

# start training
estimator.fit({'training': 's3://intonation/'})
```

You might need to adjust the variables to suit your needs.