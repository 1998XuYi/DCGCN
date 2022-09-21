import numpy as np
graph_signal_matrix_filename = r'C:\Users\xuyi\Desktop\DGCN-master\DGCN-master\data\PEMS04\pems04.npz'



data_seq = np.load(graph_signal_matrix_filename)['data']
print(data_seq.shape) # (16992,307,3)
print(data_seq)