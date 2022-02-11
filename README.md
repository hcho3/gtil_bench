# Micro-benchmark to compare GTIL and XGBoost predictor

## How to build
```bash
# Set up Conda environment
conda create -n gtil_bench python=3.8
conda activate treelite_dev
conda install -c conda-forge mamba
mamba install -c conda-forge numpy scipy matplotlib scikit-learn pandas rapidjson fmt

# Build XGBoost from source
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
mkdir build
cd build
cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
ninja -v install
cd ../..

# Build Treelite from source
git clone https://github.com/hcho3/treelite.git -b gtil_block_omp
cd treelite
mkdir build
cd build
cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
ninja -v install
cd ../..

# Build micro-benchmark
mkdir build
cd build
cmake .. -GNinja
ninja -v
cd ..
```

## Example run
```
$ ./build/gtil_bench gtil large_model.json 393 100000
GTIL
Time elapsed: 5864 ms

$ ./build/gtil_bench xgb large_model.json 393 100000
XGBoost
[13:38:09] WARNING: ../src/learner.cc:735: Found JSON model saved before XGBoost 1.6, please save the model using current version again. The support for old JSON model will be discontinued in XGBoost 2.3.
Time elapsed: 1450 ms
```
