# MLFlow Recipes in Titanic Competiton

[![GitHub](https://img.shields.io/static/v1?label=Code&message=GitHub&color=blue&style=flat-square)](https://github.com/leomaurodesenv/mlflow-recipes-titanic)
[![MIT license](https://img.shields.io/static/v1?label=License&message=MIT&color=blue&style=flat-square)](LICENSE)
   

This repository is learning code for designing a solution using MLFlow Recipes. 
[MLFlow Recipes](https://mlflow.org/docs/latest/recipes.html) is an open-source project developed by the MLFlow community to provide a set of pre-built, tested, and well-documented machine learning (ML) workflows or "recipes" that can be easily adapted to different ML tasks. These recipes are built on top of the MLFlow framework, which is an open-source platform for the complete machine learning lifecycle, including data preprocessing, model training, evaluation, and deployment.
In the project, we are going to design a solution for competition [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/).

> Reference
> - https://mlflow.org/docs/latest/recipes.html
> - https://www.kaggle.com/competitions/titanic/

---
## Code

Download or clone this repository.

### Data

1. Download the dataset in [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/)
2. Extract all the files in `./data/input/` folder
3. Now, you can run the code using `mlflow recipes`!

### Running

You can run this code locally or using [databricks](https://www.databricks.com/).

```sh
# Create a Python environment
$ python -m venv .venv
$ source .venv/bin/activate

# Install the requirements
$ pip install -r requirements.txt

# Run using: notebooks/jupyter; or
# Run using: notebooks/databricks; or
# Run using: terminal
$ mlflow recipes run --profile local

# Visualize the experiment performance
# Paths according to `profiles/local.yaml`
$ mlflow ui \
$   --backend-store-uri="sqlite:///metadata/mlflow/mlruns.db" \
$   --default-artifact-root="./metadata/mlflow/mlartifacts"
```

---
## Also look ~

- License [MIT](LICENSE)
- Created by [leomaurodesenv](https://github.com/leomaurodesenv/)
