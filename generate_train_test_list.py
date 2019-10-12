import random

'''generate train.txt and test.txt from image_list.txt'''
'''combine use with generate_imglist.py'''

train_ratio = 0.8
img_info_file = '/path/to/image_list.txt'
train_file = '/path/to/train.txt'
test_file = '/path/to/test.txt'

with open(img_info_file, 'r') as f:
    lines = [line.strip() for line in f]

random.seed(1211)
random.shuffle(lines)
train_size = int(len(lines) * train_ratio)

with open(train_file, 'w') as f:
    for line in lines[:train_size]:
        print(line, file=f)

with open(test_file, 'w') as f:
    for line in lines[train_size:]:
        print(line, file=f)

print('[Done] Test file (size={}) saves to: {}'.format(train_size, train_file))
print('[Done] Train file (size={}) saves to: {}'.format(len(lines) - train_size, test_file))

