This is a repository to implement a simple neural network by using gradient descent and back-propagation.

<img src="./images/simple_neural_network.png">

## Training the network
To training our simple neural network, please following the instructions below

```bash
git clone https://github.com/ZhongxingPeng/simple_neural_network.git
cd simple_neural_network.git
python simple_neural_network.py
```

## Set initial values
Modify the following variables in `simple_neural_network.py` to set the initial values for parameters

```python
self.lamb = 1.21463071343
self.bias = 0.971014765329
self.weight = 0.13225856384
```

which in turn will result in a following  possible output:

```bash
Epoch 0: lambda = 1.21465776706, bias = 0.971614493789, weight = 0.133034781136
Loss: 37.0323622606
Training Accu = 0.7, Test Accu = 0.4
...
...
...
Epoch 1999999: lambda = 1.30930662419, bias = 3.72328816337, weight = -1.24175243359
Loss: 6.04801669517e-05
Training Accu = 1.0, Test Accu = 1.0
```

## Search parameters for our network
Run `search_param.py` to search for better parameters. The new parameters will be appended to a file named `save_file.txt` in the current directory.

```bash
python search_param.py
```
