### PhaseDifformer: Enabling Single-Observation Decomposition of Multi-phase X-ray Diffraction Patterns via Recursive Diffusion Transformers  

# Table of Contents
- [Setup a Docker Environment](#setup-a-docker-environment)
- [Prepare Datasets](#prepare-datasets)
- [Training](#training)
- [Evaluation by synthetic data](#evaluation-by-synthetic-data)
- [Perform decomposition on custom data](#perform-decomposition-on-custom-data)


## Setup a Docker Environment
```bash
cd docker
docker build -t main/phasedifformer:latest .
docker run --gpus=all --name phasedifformer --shm-size=2g -v ../:/workspace -it  main/phasedifformer:latest /bin/bash
```
Note: If `docker run` fails due to a relative path issue, please replace `../` with the absolute path to the cloned repository directory.

## Prepare Datasets
In the docker container:
```bash
cd /workspace/Data
python query_mp.py #Download structure data from the Materials Project
python make_mp_data.py #Create a synthetic dataset for training and testing
```

## Training
In the `/workspace` directory in the docker container:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py 
```
Training is performed in the following order: noise predictor with guide, noise predictor without guide, and confidence estimator.

## Evaluation by synthetic data
In the dataset preparation described above, synthetic test datasets were constructed for single-phase, two-phase, and three-phase patterns across three crystal systems: Cubic, Tetragonal, and Orthorhombic.
By assigning the target dataset path to `path_data` and the save directory to `path_save_dir` in `recursive_decomposition.py`, and then running
```bash
CUDA_VISIBLE_DEVICES=0 python recursive_decomposition.py
```
you can perform phase separation and evaluation.

## Perform decomposition on custom data
Assign the 2θ values and intensities of the PXRD pattern as NumPy arrays to `pattern_angle` and `pattern_intensity`, respectively, in `perform_decomposition_own_data.py`, and specify the save directory in `path_save_dir`. Then, by running
```bash
CUDA_VISIBLE_DEVICES=0 python perform_decomposition_own_data.py
```
the decomposition can be performed.

By specifying the directory where the decomposition results are saved in `path_dir` and the reference structure CIF files in `path_cifs` in `show_decomposed_pattern.py`, the decomposition results can be visualized. If `path_cifs` is set to an empty list, only the decomposition results will be visualized.
