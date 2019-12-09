import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        node = nn.as_scalar(self.run(x))
        if node >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        flag = True
        while flag:
            flag = False
            batch_size = 1
            for x, y in dataset.iterate_once(batch_size):
                if self.get_prediction(x) != nn.as_scalar(y):
                    nn.Parameter.update(self.w, x, nn.as_scalar(y))
                    flag = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 1
        self.w1 = nn.Parameter(1, 100)
        self.b1 = nn.Parameter(1, 100)
        self.w2 = nn.Parameter(100, 50)
        self.b2 = nn.Parameter(1, 50)
        self.w3 = nn.Parameter(50, 1)
        self.b3 = nn.Parameter(1, 1)
        self.list = [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        relu1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        relu2 = nn.ReLU(nn.AddBias(nn.Linear(relu1, self.w2), self.b2))
        xw2 = nn.Linear(relu2, self.w3)
        return nn.AddBias(xw2, self.b3)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) > 0.01:

            for x, y in dataset.iterate_once(self.batch_size):
                grad = nn.gradients(self.get_loss(x, y), self.list)
                self.w1.update(grad[0], -0.01)
                self.w2.update(grad[1], -0.01)
                self.w3.update(grad[2], -0.01)
                self.b1.update(grad[3], -0.01)
                self.b2.update(grad[4], -0.01)
                self.b3.update(grad[5], -0.01)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 5
        self.w1 = nn.Parameter(784, 200)
        self.b1 = nn.Parameter(1, 200)
        self.w2 = nn.Parameter(200, 10)
        self.b2 = nn.Parameter(1, 10)
        self.list = [self.w1, self.w2, self.b1, self.b2]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        xw1 = nn.Linear(x, self.w1)
        biased1 = nn.AddBias(xw1, self.b1)
        relu1 = nn.ReLU(biased1)
        xw2 = nn.Linear(relu1, self.w2)
        return nn.AddBias(xw2, self.b2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while dataset.get_validation_accuracy() < 0.97:

            for x, y in dataset.iterate_once(self.batch_size):
                grad = nn.gradients(self.get_loss(x, y), self.list)
                self.w1.update(grad[0], -0.01)
                self.w2.update(grad[1], -0.01)
                self.b1.update(grad[2], -0.01)
                self.b2.update(grad[3], -0.01)

class LanguageIDModel(object):
    def __init__(self):
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        self.batch_size = 100
        self.hiddenLayerSize = 200

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.dim = 5
        self.batch_size = 2

        self.hiddendim = 400
        self.w = nn.Parameter(self.num_chars, self.hiddendim)  # w
        self.wh = nn.Parameter(self.hiddendim, self.hiddendim)  # w_hidden
        self.wf = nn.Parameter(self.hiddendim, self.dim)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.
=======
        self.w_h = nn.Parameter(self.hiddenLayerSize, self.hiddenLayerSize)
        self.w_f = nn.Parameter(self.hiddenLayerSize, len(self.languages))
        self.w = nn.Parameter(self.num_chars, self.hiddenLayerSize)


<<<<<<< HEAD
        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h = nn.Linear(xs[0], self.w)
        z = h
        for i, x in enumerate(xs[1:]):
            z = nn.Add(nn.Linear(x, self.w), nn.Linear(z, self.wh))

        return nn.Linear(z, self.wf)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while dataset.get_validation_accuracy() < 0.81:

            for x, y in dataset.iterate_once(self.batch_size):
                grad = nn.gradients(self.get_loss(x, y), [self.w, self.wh, self.wf])
                self.w.update(grad[0], -0.005)
                self.wh.update(grad[1], -0.005)
                self.wf.update(grad[2], -0.005)
    def run(self, xs):
        z = nn.Linear(xs[0], self.w)
        y = xs[1:]
        for x in y:
            z = nn.ReLU(nn.Add(nn.Linear(x, self.w), nn.Linear(z, self.wh)))
        return nn.Linear(z, self.wf)

    def get_loss(self, xs, y):
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        while dataset.get_validation_accuracy() < 0.88:
            for x in dataset.iterate_once(self.batch_size):
                grad = nn.gradients(self.get_loss(x[0], x[1]), [self.w, self.wh, self.wf])
                self.w.update(grad[0], -0.03)
                self.wh.update(grad[1], -0.03)
                self.wf.update(grad[2], -0.03)

