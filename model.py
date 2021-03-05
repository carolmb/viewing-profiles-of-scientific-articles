#!/usr/bin/env python
# coding: utf-8

import stasts

from read_file import load_data
from stasts import filter_outliers


if __name__ == '__main__':
    data = load_data('segm/segmented_curves_filtered.txt')
    data = filter_outliers(data)

    stasts.plot_life_time_hist(data, 'lifetime_v2')
    stasts.plot_no_of_visual(data, 'views_v2')
    # stasts.plot_no_of_intervals(data,'segments')


# --------------------------------------------------------------------

'''
data = load_data('r_code/segmented_curves_filtered.txt')

data = filter_outliers(data)

plot_hists(data,4,True)
plot_hists(data,5,True)
plot_hists(data,6,True)
'''





# histograma do tempo de cada artigo ok
# histograma da quantidade de quebras ok
# filtrar os artigos com 4 ou menos anos de vida ok
# histograma de visualização de artigos ok
#agrupar usando medida do paper

