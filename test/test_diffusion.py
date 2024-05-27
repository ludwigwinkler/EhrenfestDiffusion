import os

import pyrootutils

path = pyrootutils.find_root(search_from=__file__, indicator=[".git"])
pyrootutils.set_root(
    path=path,  # path to the root directory
    project_root_env_var=True,  # set the PROJECT_ROOT environment variable to root directory
    dotenv=True,  # load environment variables from .env if exists in root directory
    pythonpath=True,  # add root directory to the PYTHONPATH (helps with imports)
    cwd=True,  # change current working directory to the root directory (helps with filepaths)
)

# def test_logistic_models():
#     args = {'model.output_type': 'logistic',
#             'fast_dev_run': 2,
#             'diffusion.max_sampling_steps': 250,
#             'batch_size': 16,
#             }
#
#     run_cli_cmd = f"python main.py" + ''.join(
#         [f' --{key} {value}' for key, value in args.items()])  # construct cli command
#     exit_status = os.system(run_cli_cmd)
#     assert exit_status == 0

default_args = {'load_checkpoint': False,
                'fast_dev_run': 1,
                'diffusion.max_sampling_steps': 200,
                'batch_size': 2, }


def test_epsilon_to_score_models():
    args = {**default_args,
            'model.output_type': 'epsilon',
            'diffusion.rate': 'score',
            }

    run_cli_cmd = f"python main.py" + ''.join(
        [f' --{key} {value}' for key, value in args.items()])  # construct cli command
    exit_status = os.system(run_cli_cmd)
    assert exit_status == 0


def test_epsilon_to_taylor1_models():
    args = {**default_args,
            'model.output_type': 'epsilon',
            'diffusion.rate': 'taylor1',
            }

    run_cli_cmd = f"python main.py" + ''.join(
        [f' --{key} {value}' for key, value in args.items()])  # construct cli command
    exit_status = os.system(run_cli_cmd)
    assert exit_status == 0


def test_epsilon_to_taylor2_models():
    args = {**default_args,
            'model.output_type': 'epsilon',
            'diffusion.rate': 'taylor2',
            }

    run_cli_cmd = f"python main.py" + ''.join(
        [f' --{key} {value}' for key, value in args.items()])  # construct cli command
    exit_status = os.system(run_cli_cmd)
    assert exit_status == 0


def test_taylor1_to_taylor1_models():
    args = {'model.output_type': 'taylor1',
            'diffusion.rate': 'taylor1',
            **default_args
            }

    run_cli_cmd = f"python main.py" + ''.join(
        [f' --{key} {value}' for key, value in args.items()])  # construct cli command
    exit_status = os.system(run_cli_cmd)
    assert exit_status == 0


def test_score_to_score_models():
    args = {'model.output_type': 'score',
            'diffusion.rate': 'score',
            **default_args
            }

    run_cli_cmd = f"python main.py" + ''.join(
        [f' --{key} {value}' for key, value in args.items()])  # construct cli command
    exit_status = os.system(run_cli_cmd)
    assert exit_status == 0


def test_ratio_models():
    args = {**default_args,
            'model.output_type': 'ratio',
            'diffusion.rate': 'ratio',
            }

    run_cli_cmd = f"python main.py" + ''.join(
        [f' --{key} {value}' for key, value in args.items()])  # construct cli command
    exit_status = os.system(run_cli_cmd)
    assert exit_status == 0


def test_ratio2_models():
    args = {**default_args,
            'model.output_type': 'ratio2',
            'diffusion.rate': 'ratio2',
            }

    run_cli_cmd = f"python main.py" + ''.join(
        [f' --{key} {value}' for key, value in args.items()])  # construct cli command
    exit_status = os.system(run_cli_cmd)
    assert exit_status == 0
