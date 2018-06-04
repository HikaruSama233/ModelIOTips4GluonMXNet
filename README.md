# ModelIOTips4GluonMXNet

Small tips. 

## Load net structure (model.json) and its parameters (model.params) created by mxnet.

There will be two files, __model.json__ and __model.params__ .

If you need to use them in Gluon:

```
net = gluon.nn.SymbolBlock(
            outputs = mx.sym.load('model.json'),
            inputs = [mx.sym.Variable('data'), mx.sym.Variable('data2')]   ## It depends on the inputs of network.
        )
net.load_params('model.params', ctx = mx.cpu())  ## or mx.gpu() if you want.

## Then we can use this network like this

net(data1, data2)   ## data1 and data2 are NDArrays. They are inputs of the network.

```

## Save net structure (to json) and all its parameters created by Gluon

```
## Create dummy inputs if needed.

x = mx.sym.var('data')
x2 = mx.sym.var('data2')
y= net(x, x2)  ## net is The HybridSequential blocks
y.save('model.json')   ## save network structure to json file

model_params = net.collect_params()
model_params.save('model.params') ## save parameters

#######################################
## It is replaced by 'export' function
net.export('model')

```
