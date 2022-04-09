import sys
import csv
import random




def input_learning_constant():
    val = 0.0

    while True:
        val = float(input("Input constant between 0 and 1: "))
        if val <= 1.0 and val >= 0.0:
            break
    return val 

def csv_to_list(file_name):
    rows = []
    with open(file_name) as file:
        reader = csv.reader(file, delimiter=",")

        for row in reader:
            rows.append(row)
    return rows

def find_vector_size(matrix):
    
    size = 0
    for val in matrix:
        try:
            float(val)
            size +=1
        except:
            break
            
    return size

def create_weight_list(size):
    random.seed(3)

    matrix = []
    for i in range(size):
        matrix.append(random.uniform(0.0,1.0))
    return matrix


def update_weights(weights, learning_constant, expected_val, calculated_val, vector):

    for i in range(len(weights)):
        weights[i] = weights[i] + learning_constant*( expected_val-calculated_val )*float(vector[i])
    
    return weights

def update_deviation(deviation, learning_constant, expected_val, calculated_val):

    return deviation-learning_constant*(expected_val-calculated_val)




def main():
   
    
    alpha = input_learning_constant()
    train_list = csv_to_list("train.csv")
    test_list = csv_to_list("test.csv")
    vector_size = find_vector_size(train_list[0])
    theta = random.uniform(0.0,1.0)
    w = [0.6323841161037083, 0.0027737141178670044, 0.22507671980644678, 0.5398514648934943] #create_weight_list(vector_size)
    exp_vals = { "Iris-versicolor" : 1.0, 'Iris-virginica' : 0.0}
    acc = 0.0
    net = 0.0-theta
    

    print(w)
    print("Uczę się...")
    
    #pętla epok
    for i in range(1000):
        error = 0.0
        
        #pętla jednej pokolenia uczenia sie
        for k in range(len(train_list)):
            
            net = 0.0-theta
            y = 0.0
            d = exp_vals[train_list[k][vector_size]] #oczekiwana wartość dla nazwy gatunku key=Versicolor-> 1.0, key=virginica -> 0.0
            
            #trenowanie pojedycznego przypadku
            for j in range(vector_size):
                net += float(train_list[k][j]) * w[j]

            if net >= 0:
                y = 1.0
            
            w = update_weights(w, alpha, d,y, train_list[k])
            theta = update_deviation(theta, alpha, d, y)
            
            error += (d-y)**2
        
        error = float(error) / float(len(train_list))
        print(f'Error: {error}')
     
    print(w)
    print(f'Theta: {theta}, alfa: {alpha}')
    
    print("Testuję...")
    for i in range(len(test_list)):
            
        net = 0.0-theta
        y = 0.0
        d = exp_vals[test_list[i][vector_size]] 
        
        for j in range(vector_size):
            net += float(train_list[i][j]) * w[j]
        
        if net >= 0:
            y = 1
            
        if d == y:
            acc += 1.0

    print(f'Dokladnosc: {float(acc)/float(len(test_list))*100.0}\nAcc: {acc}')


    print(f'Wprowadz wektor {vector_size} wymiarowy w formacie "a,b,c,...x"\nType exit to exit') 
    while True:
        temp = input()
        if(temp == "exit"):
            break
        vector = str(temp).split(",")

        net = 0.0-theta
        y = 0.0

        for j in range(len(vector)):
            net += float(vector[j]) * float(w[j])
        
        if net >= 0:
            y = 1
            print("versi")
        else:
            print("virginica")
            


if __name__ == '__main__' :
    main()