import os

'''generate image list (.txt) from a folder, with format 'path_to_img label' '''
'''each sub-folder contains all images belong to one subject.'''
'''the name of the sub-foler is the label, start from 0 to N'''

f = open('generated_list.txt', 'w')
img_src = 'path_to_img_folder'
sub_dictionaries = os.listdir(img_src)

for i in range(0, len(sub_dictionaries)):
    print('%d %d\n' % (i, len(sub_dictionaries)))
    sublist = os.listdir(os.path.join(img_src, sub_dictionaries[i]))
    for j in range(0, len(sublist)):
        image_dir = os.path.join(img_src, sub_dictionaries[i], sublist[j])
        label = int(sub_dictionaries[i])
        f.write('{} {}\n'.format(image_dir, str(label)))
f.close()
