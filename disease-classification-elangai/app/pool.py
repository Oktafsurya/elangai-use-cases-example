"""
Inference pool of disease-classification-elangai application.

Generated by 'elangai generate' using `elangai` 1.1.0.
"""

from elangai import *


@elangai_pool
def inference_pool(ELANGAI_ENV):
    """Inference pool function.

    :param ELANGAI_ENV: elangai environment variable to interact with
    :return: valid dictionary for the output
    """
    return ELANGAI_ENV['disease'].predict(ELANGAI_ENV)