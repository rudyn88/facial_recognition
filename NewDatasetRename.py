# importing all packages that may be needed
import csv
import os

final_dir = 'C:/Users/lucab/Downloads/celebA/'
root_dir = 'C:/Users/lucab/Downloads/imdb/imdb_crop/imdb_crop/'
gender = 0
num = 101
if __name__ == '__main__':
    root_dir = 'C:/Users/lucab/Downloads/img_align_celeba/img_align_celeba'
    image_files = os.listdir(root_dir)
    images = []
    for filename in image_files:
        if filename.endswith('.jpg'):
            with open('C:/Users/lucab/Downloads/img_align_celeba/list_attr_celeba.csv', 'r', newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    if row[0] == filename:
                        os.rename(root_dir+'/'+filename, final_dir+'/'+'1_'+str(row[1])+'_'+str(num)+'.jpg')
                        num +=1
print('Done!')