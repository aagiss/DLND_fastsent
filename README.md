<h3>FastSent</h3>

During the <b>Deep Learning Udacity Nanodegree</b>, we were asked to implement a neuralnet with 1 hidden layer in pure Python. This project is in pure C.

<b>Code is not clean and this project is actually a playground for me to better understand concepts and technology limitations</b> but still had better metrics than LSTM/CNN approaches tried on the same dataset.

Key implementation info:
1) Pure C even for Backpropagation and Stochastic Gradient Descent
2) Multi-Threaded
3) NN Weights saved/loaded
4) Early Stopping so no need to specify number of epochs
5) Learning Rate range test so no need to specify learning rate
6) Two sentiment outputs objectivity/subjectivity and polarity (measure and score)
