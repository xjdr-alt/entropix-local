# entropix-local

This is the local version of the entropix sampler. Server version repo can be found here: [Entropix Server](https://github.com/xjdr-alt/entropix)

## Install

Clone the repo

```sh
git clone git@github.com:SinatrasC/entropix-smollm.git

cd entropix-smollm
```

### With `uv`
Install uv [here](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already (Rye installs it by default), and then:

```sh
uv venv --python 3.11

source .venv/bin/activate

uv pip install --project pyproject.toml .
```

and then:

```sh
jupyter-notebook entropix_local_torch.ipynb 
```

You could also use the model with torch directly from cli
**First and foremost you need to update the environment file with your own huggingface token.**

Currently there are 2 environment variables available:
```sh
SELECTED_MODEL_SIZE = "1B" # Selections: 1B, 3B
TOKEN = '' # Your HuggingFace token for gated model access
```

You can update the `.env.example` with your own token, and rename the file to `.env`.

Once the environment file is set and you installed and uv as above, you can invoke the model with the following:

```sh
python3 -m entr_model_torch.main --config.prompt "Which number is larger 9.11 or 9.9?" --config.stream --config.debug
```

You could also batch process prompts with the following example command:
```sh
python3 -m entr_model_torch.main --config.csv_file "prompts.csv" --config.no-stream --config.debug
```

the --help describes the cli args, here is the brief overview:

```sh

python3 -m entr_model_torch.main --help

GenerateConfig Usage:
--------------------
Required:
- prompt (str): The text prompt to generate from
    Example: --config.prompt "Once upon a time"
OR
- csv_file (str): path to csv file containing string prompts with column header 'prompts'
    Example: --config.csv_file "prompts.csv"

Optional:
- max_tokens (int): How many tokens to generate (1-2048)
    Default: 600
    Usage: --config.max_tokens 1000
- debug: Toggle debug information during generation
    Default: True
    Usage: --config.debug or --config.no-debug
- stream: Toggle output token streaming
    Default: True
    Usage: --config.stream or --config.no-stream

Example usage:
    python3 -m entr_model_torch.main --config.prompt "Which number is larger 9.11 or 9.9?" --config.stream --config.debug
    or
    python3 -m entr_model_torch.main --config.csv_file "prompts.csv" --config.no-stream --config.debug
