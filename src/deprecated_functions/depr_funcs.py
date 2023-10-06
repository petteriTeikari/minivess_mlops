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