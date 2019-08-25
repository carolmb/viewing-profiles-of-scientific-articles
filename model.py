#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from changes import plot_jumps
from hists import plot_hists
from read_file import load_data
from stats import filter_outliers

data = load_data()

data = filter_outliers(data)

# plot_life_time_hist(data,'lifetime')
# plot_no_of_visual(data,'visual')
# plot_no_of_intervals(data,'intervals')

# data_lQ2,data_geQ2 = group_by_num_visual(data)

# data, is_norm, reverse, header
# plot_jumps(data_lQ2,False,True,'imgs/original1/jumps_reverse_lQ2/')
# plot_jumps(data_lQ2,False,False,'imgs/original1/jumps_lQ2/')
# plot_jumps(data_geQ2,False,True,'imgs/original1/jumps_reverse_geQ2/')
# plot_jumps(data_geQ2,False,False,'imgs/original1/jumps_geQ2/')

# plot_hists(data,2,True)
plot_hists(data,3,True)
plot_hists(data,4,True)

# histograma do tempo de cada artigo ok
# histograma da quantidade de quebras ok
# filtrar os artigos com 4 ou menos anos de vida ok
# histograma de visualização de artigos ok
#agrupar usando medida do paper

