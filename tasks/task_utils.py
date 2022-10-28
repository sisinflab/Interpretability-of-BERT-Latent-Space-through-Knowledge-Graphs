import os, random
import tensorflow as tf
import numpy as np
import getopt, sys, yaml


def enable_determinism(seed=100):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def parse_argv_into_config(argv):

    config_file = None
    input_configuration = {}

    try:
        opts, _ = getopt.getopt(argv[1:], "c:", ["config="])
    except getopt.GetoptError:
        print("Use option --config to specify the configuration file path")
        sys.exit(1)

    for opt in opts:
        if opt[0] in ("--config", "-c"):
            config_file = opt[1]
            print(f"Loaded parameters from config file: {config_file}")
            with open(config_file) as f:
                input_configuration = yaml.load(f, Loader=yaml.FullLoader)

    if not config_file:
        raise Exception(
            "No configuration was provided (use --config options to specify one)."
        )

    return input_configuration
