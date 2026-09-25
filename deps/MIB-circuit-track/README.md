<p align="center">
    <a>
        <img src="assets/mib_logo.png", alt="MIB Splash">
    </a>
</p>

<p align="center">
    This repository provides code for the <b>circuit localization track</b> of the <a href="https://arxiv.org/abs/2504.13151">Mechanistic Interpretability Benchmark</a>, including code for circuit discovery and evaluation.
    <br/>
    <sub>
        <strong>circuits · localization · faithfulness</strong>
    </sub>
</p>

<p align="center" style="color: #cccccc">
    <a href="LICENSE.md"><img height="20px" src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="Apache 2.0"></a>
    <a href="https://huggingface.co/collections/mib-bench/mib-datasets-67f55273612ec3067a42a56b"><img height="20px" src="https://img.shields.io/badge/data-purple.svg?logo=huggingface"></a>
    <a href="https://huggingface.co/spaces/mib-bench/leaderboard"><img height="20px" src="https://img.shields.io/badge/leaderboard-red.svg?logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48IS0tIFVwbG9hZGVkIHRvOiBTVkcgUmVwbywgd3d3LnN2Z3JlcG8uY29tLCBHZW5lcmF0b3I6IFNWRyBSZXBvIE1peGVyIFRvb2xzIC0tPgo8c3ZnIHdpZHRoPSI4MDBweCIgaGVpZ2h0PSI4MDBweCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNMTUgMTlIOVYxMi41VjguNkM5IDguMjY4NjMgOS4yNjg2MyA4IDkuNiA4SDE0LjRDMTQuNzMxNCA4IDE1IDguMjY4NjMgMTUgOC42VjE0LjVWMTlaIiBzdHJva2U9IiNmNWY1ZjUiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPHBhdGggZD0iTTE1IDVIOSIgc3Ryb2tlPSIjZjVmNWY1IiBzdHJva2Utd2lkdGg9IjEuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+CjxwYXRoIGQ9Ik0yMC40IDE5SDE1VjE1LjFDMTUgMTQuNzY4NiAxNS4yNjg2IDE0LjUgMTUuNiAxNC41SDIwLjRDMjAuNzMxNCAxNC41IDIxIDE0Ljc2ODYgMjEgMTUuMVYxOC40QzIxIDE4LjczMTQgMjAuNzMxNCAxOSAyMC40IDE5WiIgc3Ryb2tlPSIjZjVmNWY1IiBzdHJva2Utd2lkdGg9IjEuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+CjxwYXRoIGQ9Ik05IDE5VjEzLjFDOSAxMi43Njg2IDguNzMxMzcgMTIuNSA4LjQgMTIuNUgzLjZDMy4yNjg2MyAxMi41IDMgMTIuNzY4NiAzIDEzLjFWMTguNEMzIDE4LjczMTQgMy4yNjg2MyAxOSAzLjYgMTlIOVoiIHN0cm9rZT0iI2Y1ZjVmNSIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4="></a>
    <a href="https://arxiv.org/abs/2504.13151"><img height="20px" src="https://img.shields.io/badge/paper-brown.svg?logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBVcGxvYWRlZCB0bzogU1ZHIFJlcG8sIHd3dy5zdmdyZXBvLmNvbSwgR2VuZXJhdG9yOiBTVkcgUmVwbyBNaXhlciBUb29scyAtLT4KPCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KPHN2ZyBmaWxsPSIjZmZmZmZmIiB2ZXJzaW9uPSIxLjEiIGlkPSJFYmVuZV8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiAKCSB3aWR0aD0iODAwcHgiIGhlaWdodD0iODAwcHgiIHZpZXdCb3g9IjAgMCA2NCA2NCIgZW5hYmxlLWJhY2tncm91bmQ9Im5ldyAwIDAgNjQgNjQiIHhtbDpzcGFjZT0icHJlc2VydmUiPgo8Zz4KCTxwYXRoIGQ9Ik01Miw2NGMyLjIwNiwwLDQtMS43OTQsNC00VjRjMC0yLjIwNi0xLjc5NC00LTQtNEgyNmMtMS42MzgsMC0zLjY2OCwwLjg0MS00LjgyOCwyTDEwLDEzLjE3MUM4Ljg0MSwxNC4zMyw4LDE2LjM2LDgsMTh2NDIKCQljMCwyLjIwNiwxLjc5NCw0LDQsNEg1MnogTTUyLDRsMCw1NmgwSDEyVjE4aDEwYzIuMjA2LDAsNC0xLjc5NCw0LTRWNEg1MnogTTIyLDYuODI4VjE0aC03LjE3MkwyMiw2LjgyOHoiLz4KCTxwYXRoIGQ9Ik0zMiwxNmgxMmMxLjEwNCwwLDItMC44OTYsMi0ycy0wLjg5Ni0yLTItMkgzMmMtMS4xMDQsMC0yLDAuODk2LTIsMlMzMC44OTYsMTYsMzIsMTZ6Ii8+Cgk8cGF0aCBkPSJNNDQsMjJIMjBjLTEuMTA0LDAtMiwwLjg5Ni0yLDJzMC44OTYsMiwyLDJoMjRjMS4xMDQsMCwyLTAuODk2LDItMlM0NS4xMDQsMjIsNDQsMjJ6Ii8+Cgk8cGF0aCBkPSJNNDQsMzJIMjBjLTEuMTA0LDAtMiwwLjg5Ni0yLDJzMC44OTYsMiwyLDJoMjRjMS4xMDQsMCwyLTAuODk2LDItMlM0NS4xMDQsMzIsNDQsMzJ6Ii8+Cgk8cGF0aCBkPSJNNDQsNDJIMjBjLTEuMTA0LDAtMiwwLjg5Ni0yLDJzMC44OTYsMiwyLDJoMjRjMS4xMDQsMCwyLTAuODk2LDItMlM0NS4xMDQsNDIsNDQsNDJ6Ii8+Cgk8cGF0aCBkPSJNNDQsNTJIMjBjLTEuMTA0LDAtMiwwLjg5Ni0yLDJzMC44OTYsMiwyLDJoMjRjMS4xMDQsMCwyLTAuODk2LDItMlM0NS4xMDQsNTIsNDQsNTJ6Ii8+CjwvZz4KPC9zdmc+"></a>
    <a href="mailto:mueller@technion.ac.il"><img height="20px" src="https://img.shields.io/badge/contact-gray.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGcgaWQ9IlNWR1JlcG9fYmdDYXJyaWVyIiBzdHJva2Utd2lkdGg9IjAiPjwvZz48ZyBpZD0iU1ZHUmVwb190cmFjZXJDYXJyaWVyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjwvZz48ZyBpZD0iU1ZHUmVwb19pY29uQ2FycmllciI+IDxwYXRoIGQ9Ik00IDcuMDAwMDVMMTAuMiAxMS42NUMxMS4yNjY3IDEyLjQ1IDEyLjczMzMgMTIuNDUgMTMuOCAxMS42NUwyMCA3IiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48L3BhdGg+IDxyZWN0IHg9IjMiIHk9IjUiIHdpZHRoPSIxOCIgaGVpZ2h0PSIxNCIgcng9IjIiIHN0cm9rZT0iI2ZmZmZmZiIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiPjwvcmVjdD4gPC9nPjwvc3ZnPg==" alt="Email"></a>
    <a href="https://mib-bench.github.io"><img height="20px" src="https://img.shields.io/badge/website-black.svg" alt="website"></a>
</p>

# Dependencies
This code requires only `EAP-IG` and `tabulate`, which is included as a submodule. You can pull submodules using `git submodule update --init --recursive`. Once it is pulled, you can install dependencies by running `pip install .` in this directory. Note that if you wish to visualize the circuits you find, you may want to `pip install EAP-IG[viz]`, which will also install the necessary `pygraphviz` package. Installing this package can be challenging, which is why it has been excluded. Our code was tested using `torch == 2.4.1`.

If you use the `oa.py` script, you will additionally need `nnsight`. We used `nnsight==0.2.15`.

# Circuit Discovery
<p align="center">
    <a>
    <img src="assets/circuit_track.png" alt="Overview of the circuit localization track."/>
    </a>
</p>

Here, we describe how to run the circuit discovery methods that we compare in the paper. In general, you can run circuit discovery by running:

```
python run_attribution.py
--models [MODELS]
--tasks [TASKS]
--method [METHOD]
--level [LEVEL="edge"]
--ablation [ABLATION="patching"]
--batch-size [BATCH_SIZE=20]
--circuit-dir [CIRCUIT-DIR="circuits/"]
```

This will iterate over each model and task specified, producing an attribution graph file for each. Each entry of `MODELS` should be in `{interpbench, gpt2, qwen2.5, gemma2, llama3}`, and each entry of `TASKS` should be in `{ioi, mcqa, arithmetic_subtraction, arithmetic_addition, arc_easy, arc_challenge}`. The `ablation` option controls the ablation used - by default patching ablations, but `mean` and `zero` ablations are also possible for certain circuit-finding methods (`eap`, `eap-ig-activations`, and `exact`). `level` is the level of granularity at which attribution is performed: `edge` (by default) or `node` / `neuron`. `batch-size` is the batch size used during attribution, and is set across models. `circuit-dir` is where circuit files are output.

We support the following attribution methods: 

- **Edge Attribution Patching (EAP; `eap`).** Note that by changing `--level` to `node` or `neuron`, you obtain node / neuron attribution patching. Node-level patching happens at the level of submodules (e.g., the MLP at layer 10, or attention head 5 at layer 3), whereas neuron-level patching assigns scores to each neuron in each of those submodules.

- **EAP with Optimal Ablations** You will first need to compute the optimal ablations vector given a model and task. This can be done by running `oa.py --models model1,models --tasks task1,task2`, which requires the `nnsight` package. Then, run `python run_attribution.py` with `--method EAP --ablation optimal --optimal_ablation_path=[PATH_to_OA_outputs]`.

- **Edge Attribution Patching with Integrated Gradients (EAP-IG; `eap-ig-inputs` / `eap-ig-activations`).**  EAP-IG-inputs runs an interpolation between many values of the input embeddings, but allows the activations to flow freely through the rest of the model from there. EAP-IG-activations interpolates between intermediate activations at the component that is being attributed. We would recommend starting with EAP-IG-inputs, as it runs faster—and, in most cases, performs better.

- **Activation Patching (`exact`).** This is the exact activation patching approach that EAP is approximating. Its runtime is long, so it is generally only feasible to run on smaller models unless you have a large enough GPU to increase the batch size significantly. Note that this approach operates at the level of edges, not nodes.

- **Information Flow Routes (IFR; `information-flow-routes`).**

- **Uniform Gradient Sampling (UGS).** To obtain the UGS results, first run [this script](https://github.com/alestolfo/optimalablation/blob/main/edge_pruning_unif_mib.py) with the `reg_lamb` hyperparameter set to 0.001. This will train the continuous mask $\alpha$ over the model’s edges. Then, run the `convert_mask_to_graph.py` script to convert the learned mask into a graph object, where each edge is assigned a weight equal to its corresponding $\alpha$ value. These edge weights are then used to determine the subgraphs for evaluation.

For example, to perform EAP-IG (inputs) with patching for IOI and MCQA on both Qwen-2.5 (0.5B) and Gemma-2 (2B) at the edge level, run:
```
python run_attribution.py \
--models qwen2.5 gemma2 \
--tasks ioi mcqa \
--method EAP-IG-inputs \
--level edge \
--ablation patching \
--batch-size 20
```

# Evaluation
To evaluate these circuits, run:
```
python run_evaluation.py
--models [MODELS]
--tasks [TASKS]
--split [SPLIT="validation"]
--level [LEVEL="edge"]
--ablation [ABLATION="patching"]
--batch-size [BATCH_SIZE=20]
--circuit-dir [CIRCUIT-DIR="circuits/"]
--output-dir [OUTPUT_DIR="results/"]
``` 
By default, this will evaluate on the validation set. To evaluate on the train or (public) test set, use `--split train` / `--split test`.

The argument structure is the same as for the attribution script, so simply port the same arguments you used when running circuit discovery while changing the python script. This will load circuits from the locations they would have been saved in when running the circuit discovery method described above.

If you are using custom circuits not obtained using this code, use the `--circuit-files` argument. This takes a series of space-separated paths to circuits to be evaluated. These circuits must be provided in either .json or .pt format; see examples provided here. 

This script will save your results in .pkl files inside `--output-dir` containing the faithfulness scores at all circuit sizes, the weighted edge counts of all circuits, and the CPR and CMD scores.

## Printing Results
Once you've finished evaluation, run:
```
python print_results.py
--output-dir [OUTPUT_DIR="results/"]
--split [SPLIT="validation"]
--metric [METRIC="cpr"]
```
This will output a table of scores for the specified split and metric. To display CMD scores instead, set `--metric cmd`. To display InterpBench scores (AUROC), use `--metric auroc`.

# Submitting to the MIB Leaderboard
If you would like to submit your circuits for evaluation on the private test set, start by collecting your circuits. We expect one folder per task/model, whre each folder contains the name of the model and the task, separated by an underscore—for example, `ioi_gpt2`, or `arc-easy_llama3`.

Each folder should contain either (1) a single .json or .pt file with floating-point importance scores assigned to each node or edge in the model, or (2) 9 .json or .pt files with binary membership variables assigned to each node or edge in the model. If (2), there should be one circuit containing no more than each of the following percentages of edges:

```{0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50}```

In other words, we expect one circuit with $k \leq 0.1$% of edges, one with $k \leq 0.2$% of edges, etc., where $k$ is the percentage of edges in the circuit compared to the full model.

We require the circuits to be publicly available on HuggingFace. We will request a URL to a directory in a HuggingFace repository that contains one folder per task/model. These folders should contain either your importance scores or your 9 circuits. If you used our code, you'll already have this directory structure: simply upload the folder corresponding to the method name.

## Example Circuits and Submissions

We provide examples of valid submissions [in this repository](https://huggingface.co/mib-bench/mib-circuits-example/tree/main). See [here](https://huggingface.co/mib-bench/mib-circuits-example/tree/main/importances/pt) for an example of importance scores, and [here](https://huggingface.co/mib-bench/mib-circuits-example/tree/main/multiple_circuits/pt) for an example of multiple circuits. You do not need to provide folders for all tasks/models; however, to prevent trivial submissions, we require you to provide circuits for $\geq$ 2 models, and $\geq$ 2 tasks.

We provide an example of an edge-level circuit in the importance-score format [here](circuits/ioi-eap-ig.json). If you choose to provide multiple circuits instead of importance scores, the circuit file format is nearly identical, but without the floating-point edge/node scores. We provide an example of a neuron-level node circuit [here](https://huggingface.co/mib-bench/mib-circuits-example/tree/main/node_examples).

## Rate Limit

There is a rate limit of 2 submissions per user per week to prevent hill-climbing on the private test set. Our automatic submission checker will verify whether what you have provided is in a valid format, and *only* count your submission toward your limit if it is. In case of issues, we ask that you provide a contact email.


## Citation
If you use the resources in this repository, please cite our [paper](https://arxiv.org/abs/2504.13151):
```
@article{mib-2025,
	title = {{MIB}: A Mechanistic Interpretability Benchmark},
	author = {Aaron Mueller and Atticus Geiger and Sarah Wiegreffe and Dana Arad and Iv{\'a}n Arcuschin and Adam Belfki and Yik Siu Chan and Jaden Fiotto-Kaufman and Tal Haklay and Michael Hanna and Jing Huang and Rohan Gupta and Yaniv Nikankin and Hadas Orgad and Nikhil Prakash and Anja Reusch and Aruna Sankaranarayanan and Shun Shao and Alessandro Stolfo and Martin Tutek and Amir Zur and David Bau and Yonatan Belinkov},
	year = {2025},
	journal = {CoRR},
	volume = {arXiv:2504.13151},
	url = {https://arxiv.org/abs/2504.13151v1}
}
```

# License
We release the content in this repository under an Apache 2.0 license.