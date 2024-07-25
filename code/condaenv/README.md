## How to use

This guide will walk you through creating the conda environment for the scripts from the `environment.yml` file and registering it as an IPython kernel.

## Step 1: Create the Conda Environment

1. **Ensure you have conda installed**: If you don't have conda installed, you can download and install Anaconda or Miniconda from the [official website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. **Create the conda environment**: Open a terminal or command prompt and navigate to this directory where the `environment.yml` file is located. Run the following command to create the environment:

   ```bash
   conda env create -f environment.yml
   ```
   
## Step 2: Activate the Conda Environment

Activate the newly created conda environment by running:

   ```bash
   conda activate rhoneCube
   ```
   
## Step 3: Install IPython Kernel
With the conda environment activated, install the IPython kernel by running:
   ```bash
    python -m ipykernel install --user --name rhoneCube --display-name "rhoneCube"
   ```


## Step 4: Verify the Kernel Installation
To ensure the kernel is installed correctly, run:
   ```bash
    jupyter kernelspec list
   ```
   
## Step 5: Using Lexcube
It might be that using Lexcube can be a little bit triccky for you when operating with Jupyter earlier than 5.2. Here you might find conflicts inside the notebook to actually find the lexcube installation.
You can check this by running:
   ```bash
   !pip show lexcube
   ```
If so, you would need to install from the notebook itself using:
   ```bash
   !pip install lexcube
   ``` 
After that it is important to install the lexcube javascript nb extension so the webapp can render the cube.
you can do so by running the following inside your activated conda environment in the command line:
   ```bash
    jupyter nbextension install --py lexcube --user
    jupyter nbextension enable --py lexcube --user
   ```

You should now be able to use all the notebooks and execute all the scripts without a problem. If one of the scripts fails, look into the corresponding sbatch file and see if you have the right configurations for the conda env to be used there.

## Updating the environment
If you update the yaml file, you can easily update your conda environment as well by running:
   ```bash
    conda env update --file environment.yml --prune
   ```