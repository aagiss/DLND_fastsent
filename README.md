<h3>FastSent</h3>

During the <b>Deep Learning Udacity Nanodegree</b>, we were asked to implement a neuralnet with 1 hidden layer in pure Python. This project is in pure C.

<b>Code is not clean and this project is actually a playground</b> but still had better metrics than some LSTM/CNN approaches tried in the same dataset.

Key implementation info:
1) Pure C even for Backpropagation and Stochastic Gradient Descent
2) Multi-Threaded
3) NN Weights saved/loaded
4) Early Stopping so no need to specify number of epochs
5) Learning Rate range test so no need to specify learning rate
6) Two sentiment outputs objevtivity/subjectivity and polarity (measure and score)
