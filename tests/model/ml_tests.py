# https://realpython.com/pytest-python-testing/
# https://towardsdatascience.com/pytest-for-machine-learning-a-simple-example-based-tutorial-a3df3c58cf8
# https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pytest/test_linear_model.py
# test_with_pytest.py

# TO-OPTIMIZE! Whether the pytest is actually needed for any of this
# This is now just a collection of the number of "pytest-like" tests in one file instead
# of being scattered around in the .py files in "model" folder


def ml_test_data_not_corrupted(corrupted_files: list):
    assert len(corrupted_files) == 0, "The following files were corrupted: {}".format(
        corrupted_files
    )
