import glob, json
import os, sys


def generate_dir():
    root_dir = '/home/alex/LabGit/ECG_random_walk/experiments/record_test/results/P-images/'
    files = glob.glob(root_dir + '*.png')
    file_names = map(lambda x: os.path.split(x)[-1], files)
    with open(root_dir + 'output.json', 'w') as fout:
        json.dump(file_names, fout)
    
def generate_dir_shortQT():
    root_dir = '/home/alex/LabGit/ECG_random_walk/experiments/record_test/results/P-images/shortQT/'
    files = glob.glob(root_dir + '*.png')
    file_names = map(lambda x: os.path.split(x)[-1], files)
    with open(root_dir + 'pnglist.json', 'w') as fout:
        json.dump(file_names, fout)

def generate_dir_infolder(folder_name):
    root_dir = '/home/alex/LabGit/ECG_random_walk/experiments/record_test/results/P-images/%s/' % folder_name
    files = glob.glob(root_dir + '*.png')
    file_names = map(lambda x: os.path.split(x)[-1], files)
    with open(root_dir + 'pnglist.json', 'w') as fout:
        json.dump(file_names, fout)
        print 'json file saved as\n %s' % (root_dir + 'pnglist.json')

if __name__ == '__main__':
    # generate_dir()
    # generate_dir_shortQT()
    generate_dir_infolder('currentImages')
