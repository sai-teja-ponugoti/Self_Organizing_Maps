'''
Created on Jun 25, 2020

@author: Sai Teja Ponugoti

If used give credits by forking, staring or watching git hub repo or in some other way. Thank you
'''

# importing required libraries
import numpy as np
import matplotlib.pyplot as plt


# function to plot heat map
def plot_heatmap(map,title_string = "default"):
    plt.figure(figsize=(5,5))
    # plt.imshow(map, cmap='hot', interpolation='nearest')
    plt.imshow(map)
    plt.title(title_string,fontweight="bold")
    plt.show()


# function to create and intialize weights matrix randomly
def create_initialize_weight_matrix(map_length, map_width, train_d, title_string="default"):
    map = np.array(np.random.rand(map_length, map_width, train_d))
    plot_heatmap(map, title_string)

    return map


def create_hij_distance_matrix(map_length, map_width):
    indexes_matrix = np.zeros((map_length, map_width, 2))
    hij_matrix_distances = np.zeros((map_length, map_width, map_width, map_width))
    for i in range(map_length):
        for j in range(map_width):
            indexes_matrix[i][j] = [i, j]

    # print(indexes_matrix.shape)
    # print(indexes_matrix[::])

    for i in range(map_length):
        for j in range(map_width):
            hij_matrix_distances[i][j] = np.linalg.norm(indexes_matrix[i][j] - indexes_matrix, axis=2) ** 2

    # print(hij_matrix_distances.shape)
    return hij_matrix_distances


def create_hij_matrix(hij_matrix_distances,sigma =1):

    hij_matrix = np.exp(-(hij_matrix_distances)/(2*(sigma**2)))

    # print(hij_matrix.shape)
    return hij_matrix


def varying_prameters_epoch(alpha,sigma, epoch_number = 1, total_epochs = 1000):
    return alpha*np.exp(-epoch_number/total_epochs),sigma*np.exp(-epoch_number/total_epochs)


# function to find best node
def find_best_matching_node(x,map):
    # print(" find_best_matching_node ")
    # print("x_shape",x.shape)
    # print("map_shape",map.shape)
    distances = np.linalg.norm(x-map,axis=2)
    # print(distances.shape)
    # amin = np.amin(distances)
    # print(amin)
    result = np.where(distances == np.amin(distances))

    return result[0][0],result[1][0]


# function to extract hij_matrix for the best winning node
def extract_hij_best_node(hij_matrix,length_id,breadth_id):
    # hij = hij_matrix[length_id,breadth_id]
    # print(hij.shape)
    # print(hij[34][58])
    return hij_matrix[length_id,breadth_id]


# function to update weights
def update_weights(map,hij,alpha,x):

    # broadcasting hij_mtrix

    hij = np.reshape(hij ,hij.shape + (1,))

    map += alpha*np.multiply(hij,(x-map))
    # print(map.shape)
    return map


# function to train/ fit the SOM map
def train(X,epochs=1000,alpha = 0.8,sigma = 1,map_length = 100, map_width = 100):

    num_features = X.shape[1]

    # randomly intialising map weights
    map = create_initialize_weight_matrix(map_length,map_width,num_features,
                                          title_string = str("randomly intialised map for sigma ="+str(sigma)))
    hij_distance_matrix = create_hij_distance_matrix(map_length,map_width)
    # print(hij_distance_matrix[0][0][99][99])

    for epoc in range(1,epochs+1):
        # caluclating epoch specific learning_Rate and sigma
        alpha_epoch, sigma_epoch = varying_prameters_epoch(alpha,sigma, epoch_number = epoc,
                                                           total_epochs = epochs)

        # calucalting sigma_epoch specific distances matrix
        hij_matrix = create_hij_matrix(hij_distance_matrix,sigma_epoch)
        # print(hij_matrix[0][0][0][0])

        # loop to go through each training sample and uppdate weights based on best matching node
        for i in range(len(X)):
            best_length,best_width = find_best_matching_node(X[i],map)
            # extract hij for best matching node
            hij_best_matching_node = extract_hij_best_node(hij_matrix,best_length,best_width)

            # update map weights based on epoch number and hij matrix for the best node
            map = update_weights(map,hij_best_matching_node,alpha_epoch,X[i])

        if epoc in [20, 40, 100, 1000]:
            # print(map)
            plot_heatmap(map,title_string = "Sigma = "+str(sigma) + "  SOM at epoch : "+ str(epoc))

        if epoc % 100 == 0:
            print("finished epoch ",epoc)


def graph(func, x_range, title_string, label_string):
   x = np.arange(*x_range)
   y = func(x)
   plt.plot(x, y, label = label_string)
   plt.title(title_string)
   plt.legend()


def plot_select_colours(scaled_colours):
    fig, ax = plt.subplots(figsize=(15, 3))

    plt.title("Selected Colours", fontweight="bold")
    ax.axis(xmin=0, xmax=len(scaled_colours))
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    for index, colour in enumerate(scaled_colours):
        ax.axvspan(index, index + 1, color=colour)


if __name__ == '__main__':

    # preparing training data
    train_data = [
        # red
        [226, 142, 142], [222, 99, 71], [239, 135, 173], [222, 115, 9], [241, 246, 100], [215, 105, 80],
        [255, 183, 168], [202, 186, 38],
        # Green
        [0, 215, 0], [80, 220, 141], [188, 220, 80], [0, 200, 156], [46, 205, 72], [66, 205, 200], [168, 205, 46],
        [170, 229, 57],
        # Blue
        [169, 169, 255], [153, 204, 255], [53, 255, 255], [180, 62, 216], [154, 117, 210], [177, 129, 192],
        [80, 168, 205], [107, 216, 223]
    ]

    train_data = np.array(train_data)
    train_data = train_data / 255

    plot_select_colours(train_data)

    #train the model using specific train data
    train(train_data, alpha=0.8, sigma=30, epochs=1000)

    # graph to show varying alpha with epoch number
    graph(lambda x: 0.8 * (np.exp(-x / 1000)), (0, 1000), "Varying Learning Rate with epochs : alpha(k)",
          "alpha = 0.8*exp(-k/T)")

    # graph to show varying sigma with epoch number , Example sigma = 1
    graph(lambda x: 1 * (np.exp(-x / 1000)), (0, 1000), "Varying Sigma with epochs : sigma(k)",
          "sigma = 1*exp(-k/T)")



# graph to show varying alpha with epoch number
graph(lambda x: 0.8*(np.exp(-x/1000)), (0,1000),"Varying Learning Rate with epochs : alpha(k)",
      "alpha = 0.8*exp(-k/T)")

# graph to show varying sigma with epoch number , Example sigma = 1
graph(lambda x: 1*(np.exp(-x/1000)), (0,1000),"Varying Sigma with epochs : sigma(k)",
      "sigma = 1*exp(-k/T)")


