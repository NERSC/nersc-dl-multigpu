# From single GPU to multiGPU training of PyTorch applications at NERSC

This repo covers material from the [Grads@NERSC](https://www.nersc.gov/events/gradsatnersc/) event. It includes minimal example scripts that show how to move from Jupyter notebooks to scripts that can run on multiple GPUs (and multiple nodes) on the Perlmutter supercomputer at NERSC. 

### Jupyter notebooks
We recommend running machine learning workloads for testing or small-scale runs on a Jupyter notebook to enable interactivity with the user. At NERSC, this can be done easily through JupyterHub with the following steps:
1. Visit [JupyterHub](https://jupyter.nersc.gov)
2. Select the required resource on Perlmutter and navigate to your local notebook. A [minimal notebook is shared here](train_single_gpu.ipynb) as an example --  most workflows will build upon these basic building blocks. The notebook defines this common workflow in PyTorch that includes
	- defining the neural network architecture [torch.nn](https://pytorch.org/docs/stable/nn.html), PyTorch [data loaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) for processing data
	- general optimization harness with a training and validation loop over the respective datasets.
3. There are in-built Jupyter kernels for machine learning that are based on the PyTorch modules (similar ones exist for the other ML frameworks as well). A quick recommendation is to use the `pytorch-2.0.1` kernel to get PyTorch 2.0 features. For building custom kernels that are based on your conda environment or other means, please see the [NERSC docs](https://docs.nersc.gov/services/jupyter/how-to-guides/#how-to-use-a-conda-environment-as-a-python-kernel). The docs are also a great resource on how to use Jupyter on Perlmutter, in general. You may also review [best practices for JupyterHub](https://www.nersc.gov/assets/Uploads/07-Jupyter-at-NERSC-Feb2024.pdf)
4. Another quick way to install your own libraries on top of what is included in the modules is to simply do `pip install --user <library_name>`. The `--user` flag will always install libraries into the path defined by the environment variable `PYTHONUSERBASE`. 
5. Quick note on libraries: While you can build your own conda environment, the other recommendation is to use modules or containers. In either case, if you need libraries in addition to what's already provided, use the `--user` flag so that the libraries are installed in `PYTHONUSERBASE`. For modules, this is defined by default and for containers, we recommend you define this variable to some local location so that user defined libraries do not interfere with the default environment.

### Notebooks to scripts
As you move to more larger workloads, the general recommendation is to use scripts -- this is especially so for multiGPU workloads, since it is tricky to get this working with Jupyter notebooks. We also recommend to group your routines into subdirectories for clean workflows. 
- The example notebook has been converted into the [`train_single_gpu` script](train_single_gpu.py) with a class structure that allows for easier extension into custom workflows.
- We have also added two additional routines that implement the checkpoint-restart function. This allows you to start the training from where a previous run ended by loading a saved model checkpoint (along with the optimizer and learning rate schedulers). We highly recommend checkpoint-restart while submitting jobs.
- To run a quick single GPU script, follow these steps:
	- Request an interactive node with `salloc --nodes 1 --qos interactive -t 30 -C gpu -A <your_account>`
	- Load a default PyTorch module for libraries with `module load pytorch/2.0.1`. You may also use your own conda environment with `module load conda; conda activate <your_env>`.
	- Run `python train_single_gpu.py`
- The script will save model checkpoints every epoch to the `outputs` directory and also the best model checkpoint that tracks the lowest validation loss (general strategy to choose the best model that avoids overfitting)


### Scripts on single GPU to multiple GPUs (and nodes)
To speed up training (due to large datasets), the most common strategy is to use data parallelism. The easiest framework here is [PyTorch DistributedDataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) or DDP, which includes comprehensive tutorials on how to use this framework. Following this, we can convert our single GPU script to multiGPU using these simple steps:
1. Initialize `torch.distributed` using
     ```
     torch.distributed.init_process_group(backend='nccl', init_method='env://')
     ```
     This will pick up the `world_size` (total number of GPUs used) and `rank` (rank of current GPU) from the environment variables that you will need to set before running -- the submit launch scripts will show you how to do that below.
 2. Set the local GPU device using the `LOCAL_RANK` environment variable (that will be defined similar to above) with
    ```
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    ```
 3. Wrap the model with `DistributedDataParallel` using
 ```
 model = DistributedDataParallel(model, device_ids=[local_rank], output_device=[local_rank])
 ```
 4. Proceed with training as you would with a single GPU. DDP will automatically sync the gradients across devices when you call `loss.backward()` during the backpropagation.
 5. Cleanup the GPU groups with `dist.destroy_process_group()`

To launch the job:
1. Define the [environment variables](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization) that allow `torch.distributed` to set up the distributed environment (`world_size` and `rank`). We implement that in [this bash script](export_DDP_vars.sh) that can be sourced when you allocate multiple GPUs using [srun](https://docs.nersc.gov/jobs/#srun). 
2. Set`$MASTER_ADDR` with `export MASTER_ADDR=$(hostname)`
 
 We implement these in our submit scripts that you can launch:
 - If you are using PyTorch modules (or your own conda env), submit [submit_batch_modules.sh](submit_batch_modules.sh) with `sbatch submit_batch_modules.sh`. Note that, in the script we have
	 - Loaded libs with `module load pytorch/2.0.1`
	 - Set up the `$MASTER_ADDR` with `export MASTER_ADDR=$(hostname)`
	 - Sourced the environment variables within the `srun` with `source export_DDP_vars.sh`: these will set the necessary variables for `torch.distributed` based on the allocated resources by `srun`. 
- For shifter containers, see [submit_batch_shifter.sh](submit_batch_shifter.sh). The commands are mostly the same except we use a containerized environment for the libraries. 
Both scripts currently submit a 2 node job, but this can be changed to any number of nodes. 

### Other best practices
- We recommend containers for more optimized libraries. NERSC provides PyTorch containers based on the [NVIDIA GPU cloud containers](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html). To query a list of PyTorch containers on Perlmutter, you can use `shifterimg images | grep pytorch`. The example shifter submit scripts use the container `nersc/pytorch:ngc-23.07-v0`. 
- Quick DDP note: checkpoint-restart needs a slight modification if you try to load a model without the `DistributedDataParallel` wrapper (for example, if you are doing inference on a single GPU) that was trained on multiple GPUs (using the `DistributedDataParallel` wrapper). 
```
		try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val 
            self.model.load_state_dict(new_state_dict)
   ``` 
 Models wrapped with DDP have an extra string `.module` that needs to be removed. The above lines in the scripts take care of this automatically
-  For logging application-specific metrics/visualizations and automatic hyperparameter optimization (HPO), we recommend [Weights & Biases](https://wandb.ai/site). See this [tutorial](https://github.com/NERSC/nersc-dl-wandb) that extends the above scripts to include Weights & Biases logging and automatic HPO on multiGPU tests.
- Before moving to data parallelism, we first recommend that you optimize your code to run on single GPUs efficiently. Check out this [in-depth tutorial](https://github.com/NERSC/sc23-dl-tutorial) that takes you step-by-step in developing a large-scale AI for Science application: this includes [single GPU optimizations and profiling](https://github.com/NERSC/sc23-dl-tutorial/tree/main?tab=readme-ov-file#single-gpu-performance-profiling-and-optimization), [data parallelism](https://github.com/NERSC/sc23-dl-tutorial/tree/main?tab=readme-ov-file#distributed-training-with-data-parallelism), and, for very large models that do not fit on a single GPU, [model parallelism](https://github.com/NERSC/sc23-dl-tutorial/tree/main?tab=readme-ov-file#model-parallelism). 
