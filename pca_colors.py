import json
import colorsys
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from read_file import breakpoints2intervals,load_data

from matplotlib import cm
from collections import defaultdict
from matplotlib.patches import Ellipse
from sklearn.utils.random import sample_without_replacement

def get_colors_pallette(N):
    HSV_tuples = [(x*1.0/N, 0.65, 0.5) for x in range(N)]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    return RGB_tuples

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def get_ellipse(v,c):
    cov = np.cov(v, rowvar=False)
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    ellipse = Ellipse((np.mean(v[:,0]),np.mean(v[:,1])),
        width=np.std(v[:,0])*2,
        height=np.std(v[:,1])*2,angle=theta,fill=False,linewidth=2.0)
    
    ellipse.set_alpha(0.4)
    ellipse.set_edgecolor(c)
    ellipse.set_facecolor(None)
    return ellipse

def plot_pca_colors(all_data,colors,filename):
    colors = np.asarray(colors)
    
    pca = PCA(n_components=2)
    pca.fit(all_data)

    y = pca.transform(all_data)

    y1_explained, y2_explained = pca.explained_variance_ratio_[:2]
    y1_explained = y1_explained*100
    y2_explained = y2_explained*100

    y1_label = 'PCA1 (%.2f%%)' % y1_explained
    y2_label = 'PCA2 (%.2f%%)' % y2_explained

    # print(y.shape)
    # if len(y) > 20000:
    #     idxs = sample_without_replacement(len(y),20000)
    #     y = y[idxs]
    #     colors = colors[idxs]
    cmap = plt.cm.get_cmap("winter")
    
    plt.figure()
    plt.scatter(y[:,0],y[:,1],alpha=0.3,c=colors,cmap=cmap)
    # plt.tight_layout()
    plt.colorbar()
    plt.xlabel(y1_label,fontsize=14)
    plt.ylabel(y2_label,fontsize=14)
    plt.title(filename)
    # plt.show()
    plt.savefig(filename+'_pca.png',bbox_inches='tight')

def select_by_N(data,N):
    dois = []
    xs_diff = []
    ys_last = []
    slopes = []
    intervals = []
    
    for i,ss,bs,xs,ys,_ in data:
        if len(ss) == N:
            dois.append(i)
            xs_diff.append(xs[-1]-xs[0])
            ys_last.append(ys[-1])
            slopes.append(np.asarray([(np.arctan(s)*57.2958) for s in ss]))
            intervals.append(np.asarray(breakpoints2intervals(bs)))

    slopes = np.asarray(slopes)
    intervals = np.asarray(intervals)
    all_data = np.concatenate((slopes,intervals),axis=1)
    return dois,xs_diff,ys_last,all_data

def get_number_of_authors(dois,data):
    number = []
    for doi in dois:
        paper = data[doi]['infos']['AU']
        number.append(len(paper.split(';')))
    return number

def get_authors_origin(paper):
    authors_origin = []
    for origin in paper['C1'].split('; ['):
        origin1 = origin.split(', ')[-1]
        if 'Peoples R China' in origin1:
            origin1 = 'China'
        if 'USA' in origin1:
            origin1 = 'USA'
        if origin1 == 'mac':
            origin1 = 'Macedonia'
        if origin1 == 'Antigua & Barbu':
            origin1 = 'W Ind Assoc St'
#         if origin == '*':
#             print(paper['C1'])
        if origin1 != '' and origin1 != '*':
            authors_origin.append(origin1)
    return authors_origin

def map_color(map_dict,authors_origin):
    counter = Counter(authors_origin)
    if len(counter) > 0:
        return counter.most_common(1)[0][0]
    else:
        return 'Unknown'

def get_colors_countries(dois,data):
    map_dict = dict()
    countries = []
    for doi in dois:
        paper = data[doi]['infos']
        authors_origin = get_authors_origin(paper)
        color = map_color(map_dict,authors_origin)
        countries.append(color)
    unique,count = np.unique(countries,return_counts=True)
    rare = set()
    for u,c in zip(unique,count):
        if c < max(count)*0.1:
            rare.add(u)
    countries = ['Others' if c in rare else c for c in countries]
    return countries

def plot_coutries(all_data,countries,filename):
    pca = PCA(n_components=2)
    pca.fit(all_data)

    y = pca.transform(all_data)

    y1_explained, y2_explained = pca.explained_variance_ratio_[:2]
    y1_explained = y1_explained*100
    y2_explained = y2_explained*100

    y1_label = 'PCA1 (%.2f%%)' % y1_explained
    y2_label = 'PCA2 (%.2f%%)' % y2_explained

    map_by_color = defaultdict(lambda:[])
    for y,c in zip(y,countries):
        map_by_color[c].append(y)
    
    plt.figure()
    
    pallette = get_colors_pallette(len(map_by_color))
    c = 0

    for k,v in map_by_color.items():
        v = np.asarray(v)
        print(len(v))
        ellipse = get_ellipse(v,pallette[c])
        ax = plt.gca()
        ax.add_artist(ellipse)
        plt.scatter(v[:,0],v[:,1],c=pallette[c],alpha=0.3,label=k)
        c += 1

    plt.legend(bbox_to_anchor=(1, 1))
    # plt.title(title+' original std='+ str(original_std[0])[:5]+' artificial std='+str(artificial_std[0])[:5])
    # plt.xlabel('original std ='+str(original_std[1])[:5]+' artificial std='+str(artificial_std[1])[:5])
    # plt.colorbar()
    plt.xlabel(y1_label,fontsize=14)
    plt.ylabel(y2_label,fontsize=14)
    plt.savefig(filename,bbox_inches='tight')
    plt.clf()
    # plt.show()

def filter(data):
    q75 = np.quantile(data,0.75)
    q25 = np.quantile(data,0.25)
    iqr = q75 - q25
    data = [min(q75+1.5*iqr,d) for d in data]
    data = [max(q25-1.5*iqr,d) for d in data]
    return data

if __name__ == "__main__":

    data = load_data()

    data_filename = 'data/papers_plos_data_time_series2_filtered.json'
    data_json = json.loads(open(data_filename,'r').read())

    for n in [2,3,4,5]:
        dois,xs_diff,ys_last,all_data = select_by_N(data,n)

        # normaliza os dados
        m = np.mean(all_data,axis=0)
        std = np.std(all_data,axis=0)
        all_data = (all_data - m)/std

        # cor por número de visualização
        xs_diff = filter(xs_diff)
        ys_last = filter(ys_last)

        
        # plot_pca_colors(all_data,ys_last,'imgs/colors_delta_visual_'+str(n))

        # plot_pca_colors(all_data,xs_diff,'imgs/colors_delta_years_'+str(n))

        #---------------------------------------------------------------------------------------

        number_of_authors = get_number_of_authors(dois,data_json)
        # plot_pca_colors(all_data,xs_diff,'imgs/colors_number_of_authors_'+str(n))

        # plt.scatter(number_of_authors,ys_last,alpha=0.6)
        # plt.xlabel('lifetime')
        # plt.ylabel('views')
        # plt.show()
        # plt.clf()

        #---------------------------------------------------------------------------------------

        countries = get_colors_countries(dois,data_json)
        plot_coutries(all_data,countries,'imgs/colors_countries_'+str(n))