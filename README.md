# **Prometheus : A Neural Network Kernel written in C and uses python wrappers for interacting with the kernel**

[doxygen code documentation](https://github.com/jar3m/n_n/docs/html/index.html)

## BUILD
To build the neural network C library that will be interface with python

make clean; make



## RUN
The python driver app is required to pass a json file containg the configurations to prometheus module

./prometheus/test.py -i examples/iris/iris.json


## Cfg file
The input config file consists of two major configurations
* `"neural_nw_lib_path"`: Path to the C shared library containing the neural network kernel
* `"neural_nw_config"`: Contains neural network configuration info
  "Config"| Description
  --------|--------------
  `"neural_nw_type"`| can either be `"REGRESS"` or `"CLASSIFY"`
  `"num_input"`| number of input
  `"num_output"`| number of output
  `"learning_rate"`| learning rate of the neural newtwork
  `"oactv"`| output activation function required only if `"neural_nw_type"` = `"CLASSIFY"`
  `"normalization"`| `"L1"` or `"L2"` `(Note curently L1 is supported L2 needs to be implemeneted)`
  `"num_hidden_layers"`| number of hidden layers and for each hidden layer define proporties
  `"hl_prop"`| list of all hidden layers properties


  "hl_prop"| list of all hidden layers properties to be define for each hiden layer
  ---------|------------------------------------
    `"actv_fn"`| `"LINEAR"`, `"RELU"`, `"SIGMOID"`
    `"size"`| size of the hidden layer ,i.e, No of neurons

* `"neural_network_test"`: Contains neural network input-output and train-test info
  Config  | Description
  --------|--------------
  `"test_file"`| Path to the file containg input-outputs used for training and testing usually a .csv or .xlsx `(Note: current support is for .csv)`
  `"delim"`| Delimiter in case of csv file
  `"inputs"`| List of label names that has to be considered as inputs `(Note: size of lablel list should be same as `"num_input"`)
  `"outputs"`| List of label names that has to be considered as outputs `(Note: size of lablel list should be same as `"num_output"`)
  `"ntrain"`| Percentage of the data to be used for training and the remaining is used for testing
  `"shuffle"`| Shuffle the data set if `1` do not shuffle if `0`
  `"epochs"`| no of training epochs `(Note: Currently only 1 is supported)`
  `"training_method"`| Training method `"Batch"` is only supported currently
  `"mean_std_scale"`| optional field mean scale the given labels
  `"min_max_scale"`| optional field min max scale the given labels
  `"stdv_scale"`| optional field standard deviation based scale the given labels
  `"hot_encode"`| optional field if Categorical variable present hot encode the outputs. more info refer `examples/iris/iris.json`


## Folders
Folder | Description
common| os includes
neural_network| Contains sources that define the neural network in C
prometheus| contains the python wrapper module
docs| contains Code documentation


## Using `prometheus` module
Below is a code snippet that shows how to use and call the prometheus module

```python
import sys, getopt
from prometheus import prometheus


p1 = prometheus()
p1.fetch_configs(sys.argv[1:])
p1.create_brain()
p1.teach_brain()
p1.sentient_brain()
p1.destroy_brain()

```

## Future Scope (Highly optimistic)
* Graphs and plots to analyse input/output
* GUI based cfg selection
* Adding Convolutional layer
* Neural network(Scaling,bounding/probalistic,)
* Training-strategy(loss Index and Optimization)
* Data-analysis(data validity and analysis)
* Model-selection(regress to get optimized model)
* Testing-analysis(How good did we do?)


## Current Scope
We have classification working but regression has some problems and seem to be beyond my scope of understanding any help will be appreciated

### Contributors
[jar3m](https://github.com/jar3m)

[kamalakannan-s](https://github.com/kamalakannan-s)

[BlindCentaur](https://github.com/BlindCentaur)

If interested in developing further join our [discord](https://discord.gg/q42YmYahpe)
