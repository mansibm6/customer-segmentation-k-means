# Customer segmentation with K-means

Plots an elbow curve and assigns mall customers to five clusters using annual income and spending score.

## Setup and execution

Use Python 3.11 and a separate virtual environment for this project. Run commands from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_inputs.py
python -m jupyterlab
```

On Windows, activate with `.venv\Scripts\activate`. For notebooks, select this environment’s kernel and execute from the top in a fresh kernel. The first code cell exposes `INPUT_PATHS` where old machine-specific paths existed. `PROJECT_DATA_DIR` changes the base for those configured paths; remaining relative paths use the notebook working directory.

## Notebooks

- [customer-segmentation-k-means.ipynb](customer-segmentation-k-means.ipynb)

## Inputs

Supply these inputs yourself; no dataset is downloaded automatically:

- `Mall_Customers.csv`

## Outputs

Elbow and customer-cluster scatter plots.

## Implementation and limitations

CSV columns at positions 3 and 4 must be annual income and spending score, as in the original Mall Customers dataset. Ten K-means initializations preserve the historical default. The centroid plot now uses the fitted five-cluster model.

## Verification

The workspace regression suite checks source/notebook validity and selected behavior with bundled or synthetic inputs. Full experiment training and historical metric reproduction are not implied. See the workspace `REFACTOR_REPORT.md` for exact test coverage and unresolved runtime requirements.

`clustering.py` separates validation, elbow calculation, and five-cluster fitting from plotting.
