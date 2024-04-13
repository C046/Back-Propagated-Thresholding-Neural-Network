A neural network from scratch, legitmately.

Default.json is a pre trained model.

I hated the fact that thresholding for the sigmoid function was constant at .5 for most models. I decided to backpropagate the thresholding using the chain rule in mathamatics via the binary cross entropy function.

Model is trained in two epochs with damn near perfection, if not perfection in accuracy considering its 1.0 rather .99, which is good depending on what ur training, this is really good.

Epoch-0-

Accuracy: 0.9257724061885155

Epoch-1-

Accuracy: 1.0
