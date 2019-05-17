#Source: https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5

from test import TestKeras


def main():
    test = TestKeras.TestKeras()
    test.explore_dataset()
    test.preproccess_data()
    test.build_model()
    test.compile_model()
    test.train_model()
    test.make_predictions()


main()


