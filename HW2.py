import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from mlxtend.plotting import plot_decision_regions
from sklearn.neighbors import KNeighborsClassifier

''' 
The Neural network (NN) implementation
* usage: Describing the architecture by list
* Hyperparameters: batch_size, epoch, learning_rate
'''
class Neural_network:
    def __init__(self, 
        batch_size: int = 32,
        epoch: int = 500,
        learning_rate: float = 0.1,
        nn_arch: list = [2,64, 64,3]):

        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.nn_arch = nn_arch

        self.input_size = nn_arch[0]
        self.output_size = nn_arch[-1]
        self.n_layer = len(nn_arch) - 1
        self.param = {}
        for i in range(self.n_layer):
            self.param[f'weight{i+1}'] = np.random.rand(nn_arch[i], nn_arch[i+1])
            self.param[f'bias{i+1}'] = np.zeros((1, nn_arch[i+1]))


    def sigmoid(self, X):
        return 1.0 / (1.0 + np.exp(-X))
    
    def sigmoid_derivative(self, X):
        return self.sigmoid(X) * (1.0 - self.sigmoid(X))

    def softmax(self, X: np.ndarray) -> np.ndarray:
        return np.exp(X) / np.sum(np.exp(X), axis=1).reshape(-1, 1)
    
    def cross_entropy(self, y_pred, y_true):
        N = y_pred.shape[0]
        loss = np.sum(-y_true * np.log(y_pred + 1e-10)) / N
        return loss

    def forward(self, X:np.ndarray, y:np.ndarray):
        self.prop = {}
        n_layer = self.n_layer
        self.feature = X
        self.label = y
        self.prop[f'a{0}'] = X

        for i in range(1, n_layer):
                self.prop[f'z{i}'] = self.prop[f'a{i-1}'] @ self.param[f'weight{i}'] + (self.param[f'bias{i}'])
                self.prop[f'a{i}'] = self.sigmoid(self.prop[f'z{i}'])
        self.prop[f'z{n_layer}'] = self.prop[f'a{n_layer-1}'] @ self.param[f'weight{n_layer}'] + (self.param[f'bias{n_layer}'])
        
        out = self.softmax(self.prop[f'z{n_layer}'])
        self.output = out
        return out 


    def backward(self):
        n_layer = self.n_layer
        N = self.label.shape[0]

        # Calculate gradients of the output layer
        self.grad = {}
        self.grad["dz"+str(n_layer)] = (self.output - self.label) / N
        self.grad["dw"+str(n_layer)] = np.dot(self.prop["a"+str(n_layer-1)].T, self.grad["dz"+str(n_layer)])
        self.grad["db"+str(n_layer)] = np.sum(self.grad["dz"+str(n_layer)], axis=0, keepdims=True)

        # Backpropagate through the hidden layers
        for i in range(n_layer-1, 0, -1):
            self.grad["da"+str(i)] = np.dot(self.grad["dz"+str(i+1)], self.param["weight"+str(i+1)].T)
            self.grad["dz"+str(i)] = self.grad["da"+str(i)] * self.sigmoid_derivative(self.prop["z"+str(i)])
            self.grad["dw"+str(i)] = np.dot(self.prop["a"+str(i-1)].T, self.grad["dz"+str(i)])
            self.grad["db"+str(i)] = np.sum(self.grad["dz"+str(i)], axis=0, keepdims=True)

        # Update the parameters
        for i in range(1, n_layer+1):
            self.param["weight"+str(i)] -= self.learning_rate * self.grad["dw"+str(i)]
            self.param["bias"+str(i)] -= self.learning_rate * self.grad["db"+str(i)]


    def split_train_val(self, x_train_feature, y_train):
        num_samples = len(x_train_feature)
        num_val_samples = int(num_samples * 0.2)
        
        # Randomly select indices for validation set
        val_indices = np.random.choice(num_samples, num_val_samples, replace=False)
        
        # Split dataset into training and validation sets
        val_feature = x_train_feature[val_indices]
        val_label = y_train[val_indices]
        train_feature = np.delete(x_train_feature, val_indices, axis=0)
        train_label = np.delete(y_train, val_indices, axis=0)
        
        return train_feature, train_label, val_feature, val_label
        

    def train(self, x_train_feature, y_train):

        train_feature, train_label, val_feature, val_label \
            = self.split_train_val(x_train_feature, y_train)
        
        train_feature, train_label = shuffle(train_feature, train_label)
        
        ## one-hot encoding
        if train_label.ndim == 1:
            train_label = np.eye(len(np.unique(train_label)), dtype=int)[train_label]
        if val_label.ndim == 1:
            val_label_onehot = np.eye(len(np.unique(val_label)), dtype=int)[val_label]
       
        feature_list = []
        label_list = []
        for j in range(max(int(train_feature.shape[0]/self.batch_size), 1)):
            batch_sample = np.random.choice(len(train_feature),
                                            self.batch_size, replace=False)
            feature_list.append(train_feature[batch_sample])
            label_list.append(train_label[batch_sample])
        
        max_iter = len(feature_list)
        
        training_loss_his = np.zeros(self.epoch)
        val_loss_his = np.zeros(self.epoch)
        val_acc_his = np.zeros(self.epoch)
        
        for i in range(1,self.epoch + 1):
            training_loss = 0
            val_loss = 0
            
            for batch_feature, batch_label in zip(feature_list, label_list):
                output = self.forward(batch_feature, batch_label)
                output_val = self.forward(val_feature, val_label_onehot)
                # loss = self.backward(output, batch_label)
                # loss_val = self.backward(output_val, val_label_onehot)
                loss = self.cross_entropy(output, batch_label)
                loss_val = self.cross_entropy(output_val, val_label_onehot)
                training_loss += loss
                val_loss += loss_val
                self.backward()

            
            training_loss_his[i - 1] = training_loss / max_iter
            val_loss_his[i - 1] = val_loss / max_iter
            val_pred = self.predict(val_feature, val_label_onehot)
            val_acc = accuracy_score(val_label, val_pred)
            val_acc_his[i - 1] = val_acc
            
            print("epoch %d/%d: validation loss: %f, Validation Accauacy: %.2f %%" 
                % (i, self.epoch, (val_loss), val_acc*100))
                
        return training_loss_his, val_loss_his, val_acc_his


    def predict(self, test_feature, test_label):
        y_pred_prob = self.forward(test_feature, test_label)
        y_pred = np.argmax(y_pred_prob, axis=1)
        return y_pred


if __name__ == "__main__":

    np.random.seed(21)
    random.seed(21)

    ''' 
    step1: Load data
    * Outputs: train_images(arr), train_labels(arr), 
                test_images(arr), train_labels(arr)
    '''
    # path
    train_dir = os.path.join('./Data_train')
    test_dir = os.path.join('./Data_test')
    labels = ['Carambula', 'Lychee', 'Pear']

    # training and test data
    train_images = []
    train_labels = []
    for label_idx, label in enumerate(labels):
        # label_dir = os.path.join(train_dir, label)
        img_dir = glob.glob(os.path.join(train_dir, label, '*.png'))
        for i, imgs in enumerate(img_dir):
            img = cv2.imread(imgs)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(gray_img, (32, 32))
            img_array = np.array(resized_img).flatten()
            train_images.append(img_array)
            train_labels.append(label_idx)
                
    test_images = []
    test_labels = []
    for label_idx, label in enumerate(labels):
        img_dir = glob.glob(os.path.join(test_dir, label, '*.png'))
        for i, imgs in enumerate(img_dir):
            img = cv2.imread(imgs)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(gray_img, (32, 32))
            img_array = np.array(resized_img).flatten()
            test_images.append(img_array)
            test_labels.append(label_idx)

    # Convert the lists to numpy arrays
    train_images = np.array(train_images) # already=1024
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    ''' 
    step2: Preprocessing and PCA
    * Outputs: train_feature, test_feature after PCA
    '''
    # Standardization
    scaler = StandardScaler()
    scaler.fit(train_images)
    train_images = scaler.transform(train_images)
    test_images = scaler.transform(test_images)

    # PCA (from 1024 to 2)
    pca = PCA(n_components=2, random_state=0)
    pca.fit(train_images)
    train_data_feature = pca.transform(train_images)    
    test_data_feature = pca.transform(test_images)

    x_train, y_train = train_data_feature, train_labels
    x_test, y_test = test_data_feature, test_labels

    ''' 
    step3: Training with Neural Network(NN) 
    * NN is defined in the class above
    * Neurons are fixed to "32"
    * Training with following hyperparameters:
        Mini-batch size, Learning rate, Number of Epoches
    '''
    nn = Neural_network()
    training_loss_his, val_loss_his, val_acc_his = nn.train(x_train, y_train) 

    ''' 
    step4: For the questions in the report:
    1. Testing accuracy
    2. Loss curve
    3. Decision reigon    
    '''
    # 1. Accuracy:
    y_pred = nn.predict(x_test, y_test)
    print("\naccuracy:", accuracy_score(y_pred, y_test)) 

    # 2. Loss curve:
    plt.figure(1)
    plt.plot(training_loss_his, label = 'training loss')
    plt.legend()
    plt.xlabel("Epoch(s)", fontsize=14)
    plt.ylabel("Cross-Entropy Loss", fontsize=14)
    plt.title("Training Loss curve")

    # 3. Decision reigon
    def plot_decision_boundary(x_test, y_test):

        pred_test = nn.predict(x_test, y_test)

        knn = KNeighborsClassifier()
        # clf2 = KNeighborsClassifier()

        knn.fit(x_test, y_test)
        # knn.fit(x_test, pred_test)

        data_path = './Data_test'
        image_class = os.listdir(data_path)

        plt.figure(2,figsize=(12,6))
        plt.subplot(121)
        ax1 = plot_decision_regions(x_test, y_test, clf=knn, legend=0)
        plt.title("The decision boundary (ground truth)")
        plt.xlabel("PCA feature 1")
        plt.ylabel("PCA feature 2")
        plt.subplot(122)
        ax2 = plot_decision_regions(x_test, pred_test, clf=knn, legend=0)
        plt.title("The decision boundary (predicted data)")
        plt.xlabel("PCA feature 1")
        plt.ylabel("PCA feature 2")
    plot_decision_boundary(x_test, y_test)
    plt.show()

    ''' 
    Comparisons with different hyperparameter settings:
    * All comparison tested with 2-layer NN
    1. Batch size
    2. Learning rate
    3. Neuron numbers    
    '''

    # def batch_size_comparison(x_train, y_train):
    #     y_pred_list=[]
    #     for i in range(4, 8):
    #         batch_size = 2 ** i
    #         nn = Neural_network(batch_size=batch_size, epoch=300, nn_arch=[2,64,64,3])
    #         training_loss_his, testing_loss_his, _ = nn.train(x_train, y_train)
    #         y_pred = nn.predict(x_test, y_test)
    #         y_pred_list.append(y_pred)
    #         plt.figure(1)
    #         plt.plot(training_loss_his, label=f'batch_size = {batch_size}')
    #         plt.title(f'Training Loss Curve of {nn.n_layer}-layer NN')
    #         plt.xlabel('Epoch(s)')
    #         plt.ylabel('Cross-entropy loss')
    #         plt.legend()

    #         plt.figure(2)
    #         plt.plot(testing_loss_his, label=f'batch_size = {batch_size}')
    #         plt.title(f'Testing Loss Curve of {nn.n_layer}-layer NN')
    #         plt.xlabel('Epoch(s)')
    #         plt.ylabel('Cross-entropy loss')
    #         plt.legend()
    #     plt.show()

    #     ### For accuracy checking
    #     # inc=4
    #     # for y_pred in y_pred_list:
    #     #     batch_size = 2 ** inc
    #     #     print("\nTesting accuracy with batch_size" + str(batch_size) +":"\
    #     #         ,accuracy_score(y_pred, y_test))
    #     #     inc = inc+1

    # batch_size_comparison(x_train, y_train)

    
    # def learning_rate_comparison(X_train, y_train):
    #     y_pred_list=[]
    #     for i in range(1, 5):
    #         learning_rate = 10 ** (-i)
    #         nn = Neural_network(learning_rate=learning_rate, epoch=300, nn_arch=[2,64,64,3])
    #         training_loss_his, testing_loss_his, _ = nn.train(x_train, y_train) 
    #         y_pred = nn.predict(x_test, y_test)
    #         y_pred_list.append(y_pred)
    #         plt.figure(1)
    #         plt.plot(training_loss_his, label=f'learning_rate = {learning_rate}')
    #         plt.title(f'Training Loss Curve of {nn.n_layer}-layer NN')
    #         plt.xlabel('Epoch(s)')
    #         plt.ylabel('Cross-entropy loss')
    #         plt.legend()
            
    #         plt.figure(2)
    #         plt.plot(testing_loss_his, label=f'learning_rate = {learning_rate}')
    #         plt.title(f'Testing Loss Curve of {nn.n_layer}-layer NN')
    #         plt.xlabel('Epoch(s)')
    #         plt.ylabel('Cross-entropy loss')
    #         plt.legend()
    #     plt.show()

    #     # ## For accuracy checking
    #     # inc = 1
    #     # for y_pred in y_pred_list:
    #     #     lr = 10**(-inc)
    #     #     print("\nTesting accuracy with lr" + str(lr) +":"\
    #     #         ,accuracy_score(y_pred, y_test))
    #     #     inc = inc+1
    # learning_rate_comparison(x_train, y_train)


    # def neurons_comparison(X_train, y_train):
    #     y_pred_list=[]
    #     for i in range(4, 8):
    #         num =  (2 ** i)
    #         nn = Neural_network(epoch=300, nn_arch=[2,num, num,3])
    #         training_loss_his, testing_loss_his, _ = nn.train(x_train, y_train) 
    #         y_pred = nn.predict(x_test, y_test)
    #         y_pred_list.append(y_pred)
    #         plt.figure(1)
    #         plt.plot(training_loss_his, label=f'neurons = {num}')
    #         plt.title(f'Training Loss Curve of {nn.n_layer}-layer NN')
    #         plt.xlabel('Epoch(s)')
    #         plt.ylabel('Cross-entropy loss')
    #         plt.legend()

    #         plt.figure(2)
    #         plt.plot(testing_loss_his, label=f'neurons = {num}')
    #         plt.title(f'Testing Loss Curve of {nn.n_layer}-layer NN')
    #         plt.xlabel('Epoch(s)')
    #         plt.ylabel('Cross-entropy loss')
    #         plt.legend()
    #     plt.show()

    #     # ## For accuracy checking
    #     # inc=4
    #     # for y_pred in y_pred_list:
    #     #     batch_size = 2 ** inc
    #     #     print("\nTesting accuracy with neurons" + str(batch_size) +":"\
    #     #         ,accuracy_score(y_pred, y_test))
    #     #     inc = inc+1
    # neurons_comparison(x_train, y_train)