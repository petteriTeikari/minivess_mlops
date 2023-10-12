from loguru import logger
import weightwatcher as ww

from src.utils.model_utils import get_nested_layers, filter_layer_names, create_pseudomodel_from_filtered_layers


def model_tests_main(model, # after epoch
                     initial_model, # model beginning of the epoch
                     test_config: dict,
                     first_epoch: bool = False):

    model_tests, model_test_metrics = {}, {}

    test_type = 'WEIGHTWATCHER'
    if test_config[test_type]['enable']:
        if first_epoch:
            logger.debug('Weightwatchers "ML Model Tests" enabled')
        # These are scalars, distributions (arrays in nomenclature of this repo)
        model_test_metrics[test_type] = weightwatchers_tests(model,
                                                             first_epoch=first_epoch)
        # Then you want to decide on the thresholds for these metrics to determine
        # whether the test failed or not, these need to be boolean values
        # or string level if we accept some "warning" for stuff that is hard to determine
        # to be boolean?
        model_tests[test_type] = {'dummy_test': True}
    else:
        if first_epoch:
            logger.debug('Skip Weightwatchers tests')

    return model_test_metrics, model_tests


def weightwatchers_tests(model,
                         layer_name_wildcards: tuple = ('conv1d', 'conv2d'),
                         first_epoch: bool = False,
                         output_dummy_output: bool = True):

    if output_dummy_output:

        # and summary dictionary of generalization metrics, debug output as a placeholder
        if first_epoch:
            logger.warning('DUMMY OUTPUT for WEIGHTWATCHERS tests')
        summary = {'log_norm': 2.11,
                   'alpha': 3.06,
                   'alpha_weighted': 2.78,
                   'log_alpha_norm': 3.21,
                   'log_spectral_norm': 0.89,
                   'stable_rank': 20.90,
                   'mp_softrank': 0.52}

    else:

        # https://github.com/CalculatedContent/WeightWatcher
        layers = get_nested_layers(model)
        conv_layers = []
        for wildcard in layer_name_wildcards:
            conv_layers += filter_layer_names(layers, layer_name_wildcard=wildcard)

        summary = None
        if len(conv_layers) > 0:
            pseudomodel = create_pseudomodel_from_filtered_layers(layers=conv_layers)
            if first_epoch:
                logger.info('{} supported convolution layers found for WeightWarcher '
                            '(note! does not support 3D convolutions atm)'.format(len(conv_layers)))
            watcher = ww.WeightWatcher(model=pseudomodel)
            details = watcher.analyze() # nothing outputted for conv2d even, TODO! examine this
            summary = watcher.get_summary(details)

        else:
            if first_epoch:
                logger.info('No convolution layers (wildcard = "{}") found for WeightWarcher '
                        '(note! does not support 3D convolutions atm)'.format(layer_name_wildcards))

    return summary