# A simple neural network that trains to convert polar coordinates to Cartesian

### Based on a description of POLARNET, in A.K. Dewdney's _The New Turing Omnibus_

---

#### Requirements (Python 3)

* `numpy` - for crunching numbers
* `colorama` - for colourising output
* `termplotlib` -  for plotting (requires [gnuplot](http://www.gnuplot.info/))

Install with pip:
`pip3 install -r requirements.txt`

---

#### Network Structure

* Input = \[radius, angle, bias\] \(Polar Coords + Bias Neuron\)
  * | Synapses1
* Medial Layer (IN) = [medin1, medin2, ..., medinN]
  * | Sigmoid Function(medin)
* Medial Layer (OUT) = [medout1, medout2, ..., medoutN]
  * | Synapses2
* Output = [x, y] (Cartesian Coords)

---

#### Configuring

* Epochs (`TRAINING_EPOCHS`) - Number of iterations to train for, lower learning rate requires more epochs to converge.
* Medial Neurons (`MEDIAL_NEURONS`) - Number of neurons per medial layer. 30-40 are recommended, more isn't always better.
* Learning Rate (`LEARNING_RATE`) - A multiplier for the amount that weights are updated each iteration, required to prevent overshoot. Smaller number may get stuck and require more epochs, but larger values may converge prematurely/sub-optimally. (e.g. 10^5 epochs for 0.01 learning rate)
* Bias (`BIAS`) - A value in an additional input neuron, to offset the others. Effectively shifts the activation functions, so that intermediate neurons aren't given a zero value. (Gives useful results across more input values.)

---

#### Debugging

Certain configuration values (e.g. LEARNING_RATE, MEDIAL_NEURONS) may give unexpected results when changed, or break some intermediate function as they grow out of range.

To make testing easier, set the random seed to some fixed value to make each test run deterministic and reproducible.
(Uncomment line starting `# seed(10)`)

---

#### Running

Simply run without any args:
`python3 polarnet.py` (or `./polarnet.py`)

![Cover](https://repository-images.githubusercontent.com/269819702/9d5fd800-ad04-11ea-8e9b-33eaecea9add)

 ---
