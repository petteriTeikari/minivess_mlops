
def log_scalars(metric_dict: dict,
                var_type: str,
                split_name: str,
                metric_type: str,
                dataset_name: str,
                epoch: int,
                cfg: dict,
                out_dir: str,
                output_artifacts: dict,
                multiple_values_per_epoch: bool = False,  # i.e. multiple batches, loss per batch
                service_name: str = 'tensorboard'):

    if len(metric_dict) > 0:
        for metric_name in metric_dict.keys():

            if metric_type == 'train':
                if 'metadata' in var_type:
                    metadata_key = '_metadata'
                else:
                    metadata_key = ''
                metric_key_out = metric_type + metadata_key + '/' + dataset_name + '/' + metric_name
            else:
                # depends what you prefer, now we want all the splits to the same subplot, add some switch if you
                # want different kind of grouping. If you have a ton of metrics this will obviously get very messy
                if 'metadata' in var_type:
                    metadata_key = 'eval_metadata/'
                else:
                    metadata_key = ''
                metric_key_out = metadata_key + metric_name + '/' + split_name + '/' + dataset_name

            if multiple_values_per_epoch:
                # you are writing a array output (array per epoch)
                no_values_per_metric = len(metric_dict[metric_name])  # e.g. number of batches per epoch
                for b in range(no_values_per_metric):
                    idx = (epoch*no_values_per_metric) + b
                    (output_artifacts['epoch_level']['tb_writer']
                     .add_scalar(metric_key_out, metric_dict[metric_name][b], idx))
            else:
                # scalar output (single value per epoch)
                (output_artifacts['epoch_level']['tb_writer']
                 .add_scalar(metric_key_out, metric_dict[metric_name], epoch))

    return output_artifacts
