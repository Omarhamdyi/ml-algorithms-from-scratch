import matplotlib.pyplot as plt

def visualize_iter(cost_history ):
        
    plt.figure()
    plt.plot(list(range(len(cost_history))), cost_history, '-r')
    plt.xlabel('iterations')
    plt.ylabel("cost")
    plt.title('cost vs iterations')
    plt.grid()
    plt.show()
