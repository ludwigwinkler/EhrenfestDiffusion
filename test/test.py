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

def test_basic():
	os.system(f"python main.py --noshow --fast_dev_run 2 --limit_val_batches 0")

def test_answer():
	assert 0 == 1