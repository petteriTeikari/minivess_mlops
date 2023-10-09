import yaml


def import_yaml_file(yaml_path: str):

    # TOADD! add scientifc notation resolver? e.g. for lr https://stackoverflow.com/a/30462009/6412152
    with open(yaml_path) as file:
        try:
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    if 'cfg' not in locals():
        raise IOError('YAML import failed! See the the line and columns above that were not parsed correctly!\n'
                      '\t\tI assume that you added or modified some entries and did something illegal there?\n'
                      '\t\tMaybe a "=" instead of ":"?\n'
                      '\t\tMaybe wrong use of "’" as the closing quote?')

    return cfg


def inference_ensemble_old():
    # DOUBLE-CHECK, why we have actually have dataset "twice", should this be removed?
    for d, dataset in enumerate(repeat_result_example):
        split_metrics[dataset] = {}
        split_metrics_stat[dataset] = {}

        for m, metric_to_track in enumerate(repeat_result_example[dataset]):
            split_metrics[dataset][metric_to_track] = {}
            split_metrics_stat[dataset][metric_to_track] = {}

            ensemble_name = get_ensemble_name(dataset_validated=dataset,
                                              metric_to_track=metric_to_track)

            model_dict = repeat_result_example[dataset][metric_to_track]['model']
            model, _, _, _ = import_model_from_path(model_path=model_dict['model_path'],
                                                    validation_config=config['config']['VALIDATION'])

            # FIXME: get this from config
            metric_dict = {'roi_size': (64, 64, 8), 'sw_batch_size': 4, 'predictor': model, 'overlap': 0.6}
            model.eval()

            with (torch.no_grad()):
                for batch_idx, batch_data in enumerate(
                        tqdm(dataloader, desc='ENSEMBLE: Inference on dataloader samples, split "{}"'.format(split),
                             position=0)):
                    inference_results = {}

                    # tqdm(repeat_results, desc='Inference on repeat', position=1, leave=False)
                    for r, repeat_name in enumerate(repeat_results):
                        sample_res = inference_sample(batch_data,
                                                      model=model,
                                                      metric_dict=metric_dict,
                                                      device=device,
                                                      auto_mixedprec=config['config']['TRAINING']['AMP'])

                        # Add different repeats together so you can get the ensemble response
                        inference_results = add_sample_results_to_ensemble_results(inf_res=deepcopy(inference_results),
                                                                                   sample_res=sample_res)

                    # We have now the inference output of each repeat in the "ensemble_results" and we can for example
                    # get the average probabilities per pixel/voxel, or do majority voting
                    # This contains the binary mask that you could actually use
                    ensemble_stat_results = ensemble_repeats(inf_res=inference_results, config=config)

                    # And let's compute the metrics from the ensembled prediction
                    sample_ensemble_metrics = (
                        get_inference_metrics(ensemble_stat_results=ensemble_stat_results,
                                              y_pred=ensemble_stat_results['arrays']['mask'],
                                              config=config,
                                              batch_data=batch_data))

                    # Collect the metrics for each sample so you can for example compute mean dice for the split
                    split_metrics[dataset][metric_to_track] = \
                        add_sample_metrics_to_split_results(sample_metrics=deepcopy(sample_ensemble_metrics),
                                                            split_metrics_tmp=split_metrics[dataset][metric_to_track])

            # Done with the dataloader here and you have metrics computed per each of the sample
            # in the dataloader and you would like to probably have like mean Dice and stdev in Dice
            # along with the individual values so you could plot some distribution and highlight the
            # poorly performing samples
            split_metrics_stat[dataset][metric_to_track] = (
                compute_split_metric_stats(split_metrics_tmp=split_metrics[dataset][metric_to_track]))

    return {'samples': split_metrics, 'stats': split_metrics_stat}



def inference_dataloader(dataloader, model, device, split: str, config: dict):

    metric_dict = {'roi_size': (64, 64, 8), 'sw_batch_size': 4, 'predictor': model, 'overlap': 0.6}
    model.eval()

    split_metrics = {}

    with (torch.no_grad()):
        for batch_idx, batch_data in enumerate(
                tqdm(dataloader, desc='BEST REPEAT: Inference on dataloader samples, split "{}"'.format(split))):

            sample_res = inference_sample(batch_data,
                                          model=model,
                                          metric_dict=metric_dict,
                                          device=device,
                                          auto_mixedprec=config['config']['TRAINING']['AMP'])

            metrics = (get_inference_metrics(y_pred=sample_res['arrays']['binary_mask'],
                                             eval_config=config['config']['VALIDATION_BEST'],
                                             batch_data=batch_data))

            split_metrics = add_sample_metrics_to_split_results(sample_metrics=deepcopy(metrics),
                                                                split_metrics_tmp=split_metrics)

    split_metrics_stat = compute_split_metric_stats(split_metrics_tmp=split_metrics)

    return split_metrics, split_metrics_stat


def log_ensembles_to_MLflow(ensemble_models_flat: dict,
                            experim_dataloaders: dict,
                            ensembled_results: dict,
                            cv_ensemble_results: dict,
                            config: dict,
                            test_loading: bool):
    """
    See these for example for logging ensemble of models to Model Registry in MLflow:
        https://www.databricks.com/blog/2021/09/21/managing-model-ensembles-with-mlflow.html
        https://medium.com/@pennyqxr/how-to-train-and-track-ensemble-models-with-mlflow-a1d2695e784b
        https://python.plainenglish.io/how-to-create-meta-model-using-mlflow-166aeb8666a8
    """
    model_uri = define_mlflow_model_uri()
    logger.info('MLflow | Model Registry model_uri = "{}"'.format(model_uri))

    mlflow_model_log = {}
    for i, ensemble_name in enumerate(ensemble_models_flat):

        logger.info('Ensemble #{}/{} | ensemble_name = {}'.
                    format(i + 1, len(ensemble_models_flat), ensemble_name))

        model_paths = ensemble_submodels
        ensemble_model_local = ModelEnsemble(models_of_ensemble=model_paths,
                                             model_best_dicts=best_dicts,
                                             models_from_paths=True,
                                             validation_config=config['config']['VALIDATION'],
                                             ensemble_params=config['config']['ENSEMBLE']['PARAMS'],
                                             validation_params=config['config']['VALIDATION']['VALIDATION_PARAMS'],
                                             device=config['config']['MACHINE']['IN_USE']['device'],
                                             eval_config=config['config']['VALIDATION_BEST'],
                                             # TODO! need to make this adaptive based on submodel
                                             precision='AMP')  # config['config']['TRAINING']['PRECISION'])

        no_submodels_per_ensemble = len(ensemble_models_flat[ensemble_name])
        mlflow_model_log[ensemble_name] = {}

        for j, submodel_name in enumerate(ensemble_models_flat[ensemble_name]):
            mlflow_model_log[ensemble_name][submodel_name] = {}

            model_path = ensemble_models_flat[ensemble_name][submodel_name]
            logger.info('Submodel #{}/{} | local_path = {}'.format(j + 1, no_submodels_per_ensemble, model_path))

            # Load the model
            model_dict = torch.load(model_path)
            model = deepcopy(model_dict['model'])
            best_dict = model_dict['best_dict']
            torch.save(dict(model=model), 'local_{}.pth'.format(submodel_name))  # debug save

            # Log the model (and register it to Model Registry)
            mlflow_model_log[ensemble_name][submodel_name] = (
                mlflow_model_logging(model=model,
                                     best_dict=best_dict,
                                     model_uri=model_uri,
                                     mlflow_config=config['config']['LOGGING']['MLFLOW'],
                                     run_params_dict=config['run'],
                                     ensemble_name=ensemble_name,
                                     submodel_name=submodel_name))

            mlflow_model_log[ensemble_name][submodel_name]['best_dict'] = best_dict

        if test_loading:
            # Test that you can download the models from the Model Registry, and that the performance
            # is exactly the same as you obtained during the training (assuming that there is no
            # stochasticity in your dataloader, like some test-time augmentation)
            logger.info('MLflow | Test that you can download model from the '
                        'Model Registry and that they are reproducible')
            test_mlflow_model_registry_load(ensemble_submodels=ensemble_models_flat[ensemble_name],
                                            mlflow_model_log=mlflow_model_log[ensemble_name],
                                            ensembled_results=ensembled_results,
                                            cv_ensemble_results=cv_ensemble_results,
                                            experim_dataloaders=experim_dataloaders,
                                            ensemble_name=ensemble_name,
                                            test_config=config['config']['LOGGING']['MLFLOW']['TEST_LOGGING'],
                                            config=config)
        else:
            logger.warning('MLflow | Skipping the model loading back from MLflow, are you sure?\n'
                           'Meant as a reproducabiloity check so that you can test that the models are loaded OK,'
                           'and give the same performance metrics as seen during the training')


def mlflow_single_model_logging(model, best_dict: dict, model_uri: str,
                                mlflow_config: dict, run_params_dict: dict,
                                ensemble_name: str, submodel_name: str):
    """
    NOTE! This does not work for meta-model
    """

    mlflow_model_log = {}
    t0 = time.time()
    artifact_name = define_artifact_name(ensemble_name, submodel_name,
                                         hyperparam_name = run_params_dict['hyperparam_name'])
    logger.info('MLflow | Logging model file to registry: {}'.format(artifact_name))

    # Log model
    # https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html#mlflow.pytorch.log_model
    # TODO! Add requirements.txt, etc. stuff around here (get requirements.txt from Dockerfile? as we have
    #  Poetry environment here
    mlflow_model_log['model_info'] = (
        mlflow.pytorch.log_model(pytorch_model=model,
                                 # registered_model_name = artifact_name,
                                 metadata={'artifact_name': artifact_name},
                                 artifact_path=artifact_name)) # Setuptools is replacing distutils.



    # Register model
    # https://mlflow.org/docs/latest/model-registry.html#adding-an-mlflow-model-to-the-model-registry
    # https://dagshub.com/blog/ml-model-registry-and-deployment-on-dagshub-with-mlflow/
    mlflow_model_log['reg_model'] = (
        mlflow.register_model(model_uri=model_uri,
                              name=artifact_name,
                              tags={'ensemble_name': ensemble_name, 'submodel_name': submodel_name}))

    logger.info('MLflow | Model log and and registering done in {:.3f} seconds'.format(time.time() - t0))

    return mlflow_model_log


def create_model_ensemble_from_mlflow_model_registry(ensemble_submodels: dict,
                                                     ensemble_name: str,
                                                     config: dict,
                                                     model_info: dict,
                                                     best_dicts: dict):
    # Define the models of the ensemble needed for the ModelEnsemble class
    models_of_ensemble = {}
    for j, submodel_name in enumerate(ensemble_submodels):
        artifact_name = define_artifact_name(ensemble_name, submodel_name,
                                             hyperparam_name=config['run']['hyperparam_name'])
        model_uri_models = f"models:/{artifact_name}/{reg_models[submodel_name].version}"
        loaded_model = deepcopy(get_model_from_mlflow_model_registry(model_uri=model_uri_models))
        torch.save(dict(model=loaded_model), '{}.pth'.format(submodel_name))  # debug save
        models_of_ensemble[submodel_name] = loaded_model

    # Create the ensembleModel class with all the submodels of the ensemble
    ensemble_model = ModelEnsemble(models_of_ensemble=models_of_ensemble,
                                   model_best_dicts=best_dicts,
                                   models_from_paths=False,
                                   validation_config=config['config']['VALIDATION'],
                                   ensemble_params=config['config']['ENSEMBLE']['PARAMS'],
                                   validation_params=config['config']['VALIDATION']['VALIDATION_PARAMS'],
                                   device=config['config']['MACHINE']['IN_USE']['device'],
                                   eval_config=config['config']['VALIDATION_BEST'],
                                   # TODO! Makle adaptive:
                                   precision='AMP')  # config['config']['TRAINING']['PRECISION'])

    return ensemble_model