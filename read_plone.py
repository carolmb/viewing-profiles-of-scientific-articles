import numpy as np

if __name__ == '__main__':
    filename = 'data/plos_one_2019.txt'

    content = open(filename,'r').read().split('\n')
    l_content = len(content) 

    data = dict()
    for idx in range(0,l_content,3):
        try:
            months = np.asarray([float(m) for m in content[idx+1].split(',')])
            views = np.asarray([float(v) for v in content[idx+2].split(',')])
            data[content[idx]] = {'months':months,'views':views}
        except:
            pass
            # artigos sem dados (annotations)

    for doi,value in data.items():
        print(doi,value)
        break
