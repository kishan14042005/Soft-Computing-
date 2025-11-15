class Neuron:
    def __init__(self, weights):
        self.weights = weights
        self.activation = 0

    def act(self, pattern):
        """Calculate the neuron's activation given an input pattern"""
        a = sum(w * p for w, p in zip(self.weights, pattern))
        return a


class Network:
    def __init__(self, w1, w2, w3, w4):
        self.neurons = [
            Neuron(w1),
            Neuron(w2),
            Neuron(w3),
            Neuron(w4)
        ]
        self.output = [0] * 4

    @staticmethod
    def threshold(value):
        return 1 if value >= 0 else 0

    def activate(self, pattern):
        for i, neuron in enumerate(self.neurons):
            print(f"\nNeuron {i} weights: {neuron.weights}")
            neuron.activation = neuron.act(pattern)
            print(f"Activation: {neuron.activation}")
            self.output[i] = self.threshold(neuron.activation)
            print(f"Output: {self.output[i]}")
        return self.output


def main():
    patrn1 = [1, 0, 1, 0]
    patrn2 = [0, 1, 0, 1]

    wt1 = [0, -3, 3, -3]
    wt2 = [-3, 0, -3, 3]
    wt3 = [3, -3, 0, -3]
    wt4 = [-3, 3, -3, 0]

    print("Hopfield Network with a single layer of 4 fully connected neurons")
    print("The network should recall the patterns 1010 and 0101 correctly.\n")

    # Create the network
    network_h = Network(wt1, wt2, wt3, wt4)

    # Test first pattern
    print("Testing pattern 1010")
    output1 = network_h.activate(patrn1)
    for i, val in enumerate(patrn1):
        if val == output1[i]:
            print(f"pattern={val}, output={output1[i]} component matches")
        else:
            print(f"pattern={val}, output={output1[i]} discrepancy occurred")

    print("\nTesting pattern 0101")
    output2 = network_h.activate(patrn2)
    for i, val in enumerate(patrn2):
        if val == output2[i]:
            print(f"pattern={val}, output={output2[i]} component matches")
        else:
            print(f"pattern={val}, output={output2[i]} discrepancy occurred")


if __name__ == "__main__":
    main()
