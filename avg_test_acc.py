import numpy as np
test_acc = []
f = open('EEGNetLog.txt', 'r')
line  = f.readline()
test_acc = line.split(" ")
test_acc = [float(x) for x in test_acc[:-1]]
avg_test_acc = np.mean(test_acc)
print('All Subject Avg Test Acc:{:.4f}'.format(avg_test_acc))    