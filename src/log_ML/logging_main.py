def save_models_if_improved(best_dict, train_epoch_results, eval_epoch_results, config):

    a = 1

def init_best_dict_with_criteria():

    # placeholder, e.g. you could have multiple metrics that you want to track, and for multiple different
    # datasets that you might have in your val_loaders, e.g. Dice/Hausdorff for both Minivess and some another dataset
    # would give you 4 different "best models" to disk (especially nice if you have some disagreement let's say with
    # your colleagues or literature, what is actually the best metric to track -> just track all possibilities)
    best_dict = {'dice/MINIVESS': {}}

    return best_dict


def log_epoch_results(train_epoch_results, eval_epoch_results, epoch, config):
    a = 1

def log_n_epochs_results(train_results, eval_results, best_dict, config):
    a = 1

def log_ensemble_results():
    a = 1