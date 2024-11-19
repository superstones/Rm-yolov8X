from random import random

import comet_ml
from ultralytics import YOLO
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
experiment = Experiment(
    api_key="XXessZE9oJUVrgeA3zyuLeWKB",
    project_name="Rm-yolov8X",
    workspace="292454993-qq-com"
)


if __name__ == '__main__':
    # Initialize the YOLO model
    model = YOLO('ultralytics/cfg/models/v8/my_yolov8x.yaml')

    # Tune hyperparameters on COCO8 for 30 epochs
    model.tune(data='Argoverse.yaml', epochs=20, iterations=1000, optimizer='AdamW', plots=True, save=True)
    comet_ml.login()
    exp = comet_ml.start()

    for step in range(0, 1000):
        # Create an example metric
        my_metric = random.random()
        # Log the metric to Comet
        exp.log_metric(name="my_metric", value=my_metric, step=step)
    # Report multiple hyperparameters using a dictionary:
    # hyper_params = {
    #     "learning_rate": 0.01,
    #     "steps": 100000,
    #     "batch_size": 50,
    # }
    # experiment.log_parameters(hyper_params)
    #
    # # Initialize and train your model
    # # model = TheModelClass()
    # # train(model)
    #
    # # Seamlessly log your Pytorch model
    # log_model(experiment, model, model_name="yolov8n")
