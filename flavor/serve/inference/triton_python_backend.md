# Python Backend for Triton Inference Server

Official Github: https://github.com/triton-inference-server/python_backend

## Folder Structure

- All filenames must match the example below, except for those in `<...>`
```
.
├── <version>
│   ├── model.py
├── config.pbtxt
├── (optional) <env>.tar.gz
└── (optional) triton_python_backend_stub
```
- [`<version>/model.py`](#modelpy): python backend model implementation
- [`config.pbtxt`](#configpbtxt): model definition for triton inference server
- [(optional) `<env>.tar.gz`](#optional-envtargz): python environment
- [(optional) `triton_python_backend_stub`](#optional-triton_python_backend_stub): python runtime

### `model.py`

Inherite `BaseTritonPythonModel` to simplify development.

### `config.pbtxt`

See [doc](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html) for detail. 

- Set `backend` field to `python`.
- Do not set `platform` field.

### (optional) `triton_python_backend_stub`

**If you need a different Python version.**

Official doc: https://github.com/triton-inference-server/python_backend?tab=readme-ov-file#building-custom-python-backend-stub

You only need to compile a Python backend stub if the Python version **does not match** that on triton inference server. In `nvcr.io/nvidia/tritonserver:24.05-py3`, the Python version is `3.10`. You would only need to compile a stub if you do not use `python3.10`.

1. Use `tritonserver` docker image: `nvcr.io/nvidia/tritonserver:24.05-py3`

    You can change image version to whichever you like. Beware that different image version may come with different Python version. You only need to compile the stub if your Python version does not match the default Python version.

2. Install software packages

    ```
    apt install cmake rapidjson-dev libarchive-dev 
    ```

3. (optional) Prepare the Python version

    If you use virtual environment, you must activate it first.

4. Build Python Backend stub


    ```
    BRANCH=r24.05

    git clone https://github.com/triton-inference-server/python_backend -b $BRANCH
    cd python_backend
    
    mkdir build && cd build
    
    cmake -DTRITON_ENABLE_GPU=ON \
    -DTRITON_BACKEND_REPO_TAG=$BRANCH \
    -DTRITON_COMMON_REPO_TAG=$BRANCH \
    -DTRITON_CORE_REPO_TAG=$BRANCH \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..

    make triton-python-backend-stub
    ```

5. Check

    ```
    ldd triton_python_backend_stub
    ```
    You would see something like `libpython3.9.so.1.0 => /root/miniconda3/envs/python3.9/lib/libpython3.9.so.1.0` which should match your Python version.

6. Move `triton_python_backend_stub` to your target model repository.

### (optional) `<env>.tar.gz`

**If you want to build your own environment.**

Official doc: https://github.com/triton-inference-server/python_backend?tab=readme-ov-file#creating-custom-execution-environments

1. Install conda

    ```
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ```

2. Create new `conda` environment and activate

    Make sure the conda env uses the correct Python version.
    If you build your own `triton_python_backend_stub`, it should match the conda Python version. Otherwise, use the same Python version as the default one in `tritonserver` docker image.
    ```
    ~/miniconda3/bin/conda create --name python3.9 python=3.9
    source ~/miniconda3/bin/activate python3.9
    ```

3. Install packages

    ```
    export PYTHONNOUSERSITE=True

    pip3 install torch==2.1.0 torchvision==0.16.0
    pip3 install numpy==1.26
    ```

4. Install packing tools

    ```
    pip3 install conda-pack
    ```

    Install additional packages to avoid `GLIBCXX_3.4.30` error:
    ```
    conda install -c conda-forge libstdcxx-ng=12 -y
    ```

5. Pack conda environment

    ```
    conda-pack -f --ignore-missing-files --exclude lib/python3.1
    ```

    Move the packed `.tar.gz` file to model repository.

6. Update the `config.pbtxt`

    Add this to model configuration:
    ```
    parameters: {
        key: "EXECUTION_ENV_PATH",
        value: {string_value: "$$TRITON_MODEL_DIRECTORY/python3.9.tar.gz"}
    }
    ```