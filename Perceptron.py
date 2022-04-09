class Perceptron:

    def __init__(self, train, test, alpha,theta,vector_size, w, exp_val):
        self.train = train
        self.test = test
        self.alpha = alpha
        self.theta = theta
        self.vector_size = vector_size
        # "w" stands for weight vector
        self.w = w
        self.exp_val = exp_val
        self.convert_to_floats()
    def convert_to_floats(self):
        
        for i in range(len(self.train)):
            for j in range(self.vector_size):
                self.train[i][j] = float(self.train[i][j])

        for i in range(len(self.test)):
            for j in range(self.vector_size):
                self.test[i][j] = float(self.test[i][j])

    def train_model(self, epochs):
        error_list= []
        for i in range(epochs):
            
            error = 0.0
            for j in range(len(self.train)):
                net = 0.0 - self.theta
                y = 0.0
                d = self.exp_val[self.train[j][self.vector_size]]

                for k in range(self.vector_size):
                    net += self.train[j][k] * self.w[k]
                

                if net >= 0.0:
                    y = 1.0

                self.update_weights(d,y,j,k)
                self.update_theta(d,y)
                error += (d-y)**2
            error = float(error) / float(len(self.train))
            error_list.append(error)

        print(f'Min error: {min(error_list)}\nMax error: {max(error_list)}')

    def test_model(self):
        acc = 0
        for i in range(len(self.test)):
            net = 0.0 - self.theta
            y = 0.0
            d = self.exp_val[self.test[i][self.vector_size]]

            for j in range(self.vector_size):
                net += self.test[i][j] * self.w[j]
            
            if net >= 0.0:
                y = 1.0
            
            if d==y:
                acc +=1
        
        print(f'Hits: {acc} - {100.0*float(acc)/float(len(self.test))}%\n')

    def test_single_vector(self):
        nums = input(f'Provide {self.vector_size} dimensional vector in a,b,c,d..z format:\n')
        nums = str(nums).split(',')

        for i in range(len(nums)):
            nums[i] = float(nums[i])

        net = 0.0-self.theta
        y = 0.0

        for j in range(len(nums)):
            net += nums[j]* self.w[j]
        
        if net >= 0:
            print("Iris-versicolor")
        else:
            print("Iris-virginica")


    def update_weights(self,d,y,j,k):

        for i in range(self.vector_size):
            self.w[i] += self.alpha*(d-y)*self.train[j][k]          

    def update_theta(self,d,y):
        self.theta -=  self.alpha*(d-y)         
