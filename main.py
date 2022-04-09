import csv
import sys
import random
import Perceptron as p


def csv_to_list(name):

    data = []

    with open(name) as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

def get_vector_size(data_set):
    return len(data_set[0]) - 1

def set_learning_constant():
    return float(input("Provide the float learning constant: "))

def create_weight_vector(vector_size):

    vector = []

    for i in range(vector_size):
        vector.append(random.random())
    
    return vector


def main():
    train = csv_to_list("train.csv")
    test = csv_to_list("test.csv")
    alpha = set_learning_constant()
    theta = random.random()
    vector_size = get_vector_size(train)
    w = create_weight_vector(vector_size)
    exp_val = { 'Iris-versicolor' : 1.0, 'Iris-virginica' : 0.0}

    perc = p.Perceptron(train,test,alpha,theta,vector_size,w,exp_val)
    perc.train_model(100)
    perc.test_model()
    perc.train_model(1000)
    perc.test_model()
    perc.train_model(10000)
    perc.test_model()
    perc.test_single_vector()


if __name__ == '__main__':
    main()