

import csv
import matplotlib.pyplot as plt

def load_vals(filename):

    file = open(filename)
    data = [row for row in csv.reader(file)]
    file.close()

    if len(data) > 1:
        print("Problem with file", filename)
        exit()
    
    ret_val = [ float(val) for val in data[0] ]

    return ret_val


def main():
    val_acc_file    = 'val_accs.txt'
    train_acc_file  = 'train_accs.txt'
    val_lss_file    = 'val_loss.txt'
    train_lss_file  = 'train_loss.txt'

    
    val_accs = load_vals(val_acc_file)
    train_accs = load_vals(train_acc_file)
    val_loss = load_vals(val_lss_file)
    train_loss = load_vals(train_lss_file)
    epochs   = [i+1 for i in range(len(val_accs))]

    plt.plot(epochs, val_accs, 'g')
    plt.plot(epochs, train_accs, 'b')
    plt.legend(['Validation Accuracy', 'Training Accuracy'], loc='upper left')
    plt.ylabel('Accuracy %')
    plt.xlabel('Epochs')
    plt.ylim(45,80)
    plt.show()

    plt.plot(epochs, val_loss, 'g')
    plt.plot(epochs, train_loss, 'b')
    plt.legend(['Validation Loss', 'Training Loss'], loc='upper right')
    plt.ylabel('Loss %')
    plt.xlabel('Epochs')
    plt.ylim(45,70)
    plt.show()

    
    

main()