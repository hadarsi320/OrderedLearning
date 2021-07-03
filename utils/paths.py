import socket

if socket.gethostname() == 'Hadars_Laptop':
    save_dir = r'C:\Users\Hadar\PycharmProjects\OrderedLearning\saves'
    datasets_dir = r'C:\Users\Hadar\PycharmProjects\OrderedLearning\datasets'
    imagenette_train_dir = r'C:\Users\Hadar\PycharmProjects\OrderedLearning\datasets\imagenette2-320/train/'
    imagenette_val_dir = r'C:\Users\Hadar\PycharmProjects\OrderedLearning\datasets\imagenette2-320/val/'

else:
    save_dir = '/mnt/ml-srv1/home/hadarsi/ordered_learning/saves/'
    datasets_dir = '/mnt/ml-srv1/home/hadarsi/ordered_learning/datasets/'
    imagenette_train_dir = '/mnt/ml-srv1/home/hadarsi/ordered_learning/datasets/imagenette2-320/train/'
    imagenette_val_dir = '/mnt/ml-srv1/home/hadarsi/ordered_learning/datasets/imagenette2-320/val/'
