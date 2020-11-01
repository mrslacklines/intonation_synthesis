import boto3
import sagemaker as sage
from sagemaker import get_execution_role

####

role = get_execution_role()
sess = sage.Session()

account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name  # or setup the region by hand
image = '{}.dkr.ecr.{}.amazonaws.com/intonation-vd'.format(account, region)

####

estimator = sage.estimator.Estimator(
    image,
    role, 1,
    'ml.p3.16xlarge',
    train_volume_size=200,
    output_path="s3://{}/output".format(sess.default_bucket()),
    sagemaker_session=sess,
    hyperparameters={
        "epochs": 2,
        "learning_rate": 0.1,
        "num_layers": 2,
        "hidden_size": 1297,
        "dropout": 0.2,
        "batch_size": 4,
        "dataset_size_limit": False,
        "cpu_count": 0
    },
    metric_definitions=[
        {'Name': 'loss', 'Regex': 'LOSS: (.*?);'}
    ]
)


####


# start training
estimator.fit({'training': 's3://intonation-vd/'}, wait=True)
