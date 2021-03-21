# Explainable Deep Neural F0 Model of Polish Read Speech
This projects aims at training a deep neural network for modelling Polish read speech F0 and exploring the relevance of input linguistic features on the predicted F0 contours.
Currently we are using AWS SageMaker as the cloud service for running both the training and inference. The generation of feature relevance and plotting can be run both in cloud and locally.
This project is the main part of my PhD dissertation, which I (**Tomasz Kuczmarski**) am currently writing up. Keep your fingers crossed :).

## Build containers
Build and push a docker container to the Amazon Web Services ECR repository
`./build_and_push.sh`

or 

`docker build -f Dockerfile-dev . -t intonation-dev`
to build for local development.

## Run training locally
In order to run the training container locally you need to mount you dataset to the docker container to emulate the SageMaker environment, i.e.

If your data lives in `/media/tomaszk/DANE/Speech_archive/HTS-demo_AMU_PL_ILO_STRAIGHT` on the host machine like mine does, you can simply:

`docker run -it --ipc="private" --shm-size=8000000000 -v /media/tomaszk/DANE/Speech_archive/HTS-demo_AMU_PL_ILO_STRAIGHT:/opt/ml/input/data/training -v /media/tomaszk/DANE/workspace/intonation:/opt/ml/model intonation-dev`

## Run training on AWS SageMaker

Open up your Notebook instance and run:
```
import boto3
import sagemaker as sage
from sagemaker import get_execution_role

####

role = get_execution_role()
sess = sage.Session()

account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name  # or setup the region by hand
image = '{}.dkr.ecr.{}.amazonaws.com/intonation-dev'.format(account, region)

####

estimator = sage.estimator.Estimator(
    image,
    role, 1,
    'ml.p3.16xlarge',
    train_volume_size=200,
    output_path="s3://{}/output".format(sess.default_bucket()),
    sagemaker_session=sess,
    hyperparameters={
        "epochs": 200,
        "learning_rate": 0.1,
        "num_layers": 2,
        "hidden_size": 64,
        "dropout": 0.2,
        "batch_size": 8,
        "dataset_size_limit": false,
        "cpu_count": 0
    },
    metric_definitions=[
        {'Name': 'loss', 'Regex': 'LOSS: (.*?);'}
    ]
)


####


# start training
estimator.fit({'training': 's3://intonation-dev/'}, wait=True)

```

You might need to adjust the variables to suit your needs.


## Generating feature relevance reports and plots
This part still needs documenting. For the time being please inspect the `create_plots.py` and `Dockerfile-plots` files. These are run in the same manner as the training part. This part generates rich results including LRP.Z analysis plots for individual files, different types of mean LRP plots for the whole dataset and a number of CSV files with feature relevance rankings and prediction MSE calculations. I promise I will document those when I finish writing the paper :).
The results are also available under `results/` in this repository. Below you can find some examples of what kind of data can be generated with the current code.

# Results

## Individual results
The generated results include invidual analyses for each of the testset recordings. Each analysis comprises of a Layer-wise Relevance Propagation result heatmap aligned with the actual fundamental frequency plotted against the predicted values. The title of each plot is an ortographic transcription of the utterance, i.e.:
![amu_pl_ilo_BAZA_2006A_zbitki_A0119](https://github.com/mrslacklines/intonation_synthesis/blob/master/intonation_synthesis/results/plots/individual/amu_pl_ilo_BAZA_2006A_zbitki_A0119.png)

Additionally, the predicted F0 is plotted without log-normalization.
![amu_pl_ilo_BAZA_2006A_zbitki_A0119_simple_pred_freq](https://github.com/mrslacklines/intonation_synthesis/blob/master/intonation_synthesis/results/plots/individual/amu_pl_ilo_BAZA_2006A_zbitki_A0119_simple_pred_freq.png)

## General results
Additionally, the script calculates different kinds of summed and mean relevance arrays for all of the 191 speech samples in the testset.
* Sum
![sum](https://github.com/mrslacklines/intonation_synthesis/blob/master/intonation_synthesis/results/plots/sum.png)
* Absolute sum
![abs sum](https://github.com/mrslacklines/intonation_synthesis/blob/master/intonation_synthesis/results/plots/absolute_sum.png)
* Positive values-only sum
![pos sum](https://github.com/mrslacklines/intonation_synthesis/blob/master/intonation_synthesis/results/plots/positive_values_sum.png)
* Negative values-only sum
![neg sum](https://github.com/mrslacklines/intonation_synthesis/blob/master/intonation_synthesis/results/plots/negative_values_sum.png)
* Mean
![mean](https://github.com/mrslacklines/intonation_synthesis/blob/master/intonation_synthesis/results/plots/mean_(sum).png)
* Mean of absolute sum
![mean abs sum](https://github.com/mrslacklines/intonation_synthesis/blob/master/intonation_synthesis/results/plots/mean_(absolute_sum).png)
* Mean of positive values-only sum
![mean pos sum](https://github.com/mrslacklines/intonation_synthesis/blob/master/intonation_synthesis/results/plots/mean_(positive_only_sum).png)
* Mean of negative values-only sum
![mean neg sum](https://github.com/mrslacklines/intonation_synthesis/blob/master/intonation_synthesis/results/plots/mean_(negative_only_sum).png)

### Feature group relevance
Features were grouped into groups on three levels of granularity and their various means and standard deviations where plotted. Since the single most relevant feature (Voiced/Unvoiced or`VUV`) flattened the plots of the least relevant ones additional plots excluding the VUV were also added.
The following examples show only the feature group means with and without VUVs. The rest of the plots can be found in the `results/plots/` directory of this repository.

* Low granularity groups
![general mean feats relevance](https://github.com/mrslacklines/intonation_synthesis/blob/master/intonation_synthesis/results/plots/feature_relevance_ranking_-_mean_(sum)_-_general.png)
![general mean feats relevance](https://github.com/mrslacklines/intonation_synthesis/blob/master/intonation_synthesis/results/plots/feature_relevance_ranking_-_mean_(sum)_-_general_-_no_vuv.png)


* Medium granularity groups
![medium mean feats relevance](https://github.com/mrslacklines/intonation_synthesis/blob/master/intonation_synthesis/results/plots/feature_relevance_ranking_-_mean_(sum)_-_detailed.png)
![medium mean feats relevance](https://github.com/mrslacklines/intonation_synthesis/blob/master/intonation_synthesis/results/plots/feature_relevance_ranking_-_mean_(sum)_-_detailed_-_no_vuv.png)


* High granularity groups
![all mean feats relevance](https://github.com/mrslacklines/intonation_synthesis/blob/master/intonation_synthesis/results/plots/feature_relevance_ranking_-_mean_(sum)_-_all.png)
![all mean feats relevance](https://github.com/mrslacklines/intonation_synthesis/blob/master/intonation_synthesis/results/plots/feature_relevance_ranking_-_mean_(sum)_-_all_-_no_vuv.png)
