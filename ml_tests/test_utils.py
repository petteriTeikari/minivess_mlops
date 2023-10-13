from loguru import logger

def add_boolean_and_metric_strings_to_summary(metrics: dict,
                                              booleans: dict,
                                              report_string: str,
                                              base_tab: str = '\t\t\t',
                                              all_tests_ok: bool = True,
                                              params_to_print: tuple = None):

    for test_name in metrics:
        test_boolean = booleans[test_name]
        report_string += f'{base_tab}{test_name}: {test_boolean}\n'
        test_metrics = metrics[test_name]
        for problem_sample in test_metrics:
            problem_dict = test_metrics[problem_sample]
            report_string += f'{base_tab}\t{problem_sample}\n'
            for key in params_to_print:
                try:
                    report_string += f'{base_tab}\t\t{key} = {problem_dict[key]}\n'
                except:
                    logger.warning('cannot find desired key = "{}" in the problem_dict'.format(key))
        all_tests_ok = all_tests_ok and test_boolean

    return report_string, all_tests_ok