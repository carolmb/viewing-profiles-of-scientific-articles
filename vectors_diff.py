from read_file import load_data
import matplotlib.pyplot as plt
import pandas as pd

def get_orig_data():
	f_segm = 'segm/segmented_curves_ori_original.txt'
	content_segm = open(f_segm,'r').read().split('\n')

	data = load_data()

	data_dict = dict()
	for i,s,b,xs,ys,p in data:
		data_dict[i] = (xs,ys)

	data_segm_dict = dict()

	j = 0
	N = len(content_segm)-1
	for i in range(0,N,4):
		data_segm_dict[content_segm[i]] = [float(m) for m in content_segm[i+2].split(' ')]
		j += 1
	print('numero de dados originais',j)
	return data_dict,data_segm_dict

def get_syn_data(mse):
	f_segm = 'segm/segmented_curves_syn_original.txt'
	content_segm = open(f_segm,'r').read().split('\n')

	data_segm_dict = dict()

	N = len(content_segm)-1
	j = 0
	for i in range(0,N,4):
		data_segm_dict[content_segm[i]] = [float(m) for m in content_segm[i+2].split(' ')]
		j += 1

	print('numero de dados sinteticos',j)

	f_data = 'segm/syn_data.txt'
	content_data = open(f_data,'r').read().split('\n')
	data_dict = dict()
	

	N = len(content_data)-1
	for i in range(0,N,3):
		x = [float(m) for m in content_data[i+1].split(',')]
		y = [float(m) for m in content_data[i+2].split(',')]
		# if mse[j] < 0.0001:
		data_dict[content_data[i]] = (x,y)
	
	return data_dict,data_segm_dict

def get_X_Y(data_dict,data_segm_dict,k):
	X = []
	Y = []
	i = 0
	for doi,ls in data_segm_dict.items():
		try:
			if len(ls) != k-1: # duas quebras, três intervalos
				continue

			xs,ys = data_dict[doi]

			delta_l = [xs[-1]-xs[0]]
			delta_v = [ys[-1]-ys[0]]

			# print(delta_l)
			# print(delta_v)

			# delta_l = [ls[0]-xs[0]]
			# for i in range(len(ls)-1):
			# 	delta_l.append(ls[i+1]-ls[i])
			# delta_l.append(xs[-1]-ls[-1])
			

			# delta_v = []
			# for l in ls:
			# 	for x,y in zip(xs,ys):
			# 		if x > l:
			# 			if len(delta_v) == 0:
			# 				delta_v.append(y)
			# 			else:
			# 				delta_v.append(y-delta_v[-1])
			# 			break
			# delta_v.append(ys[-1]-delta_v[-1])

			X += delta_l
			Y += delta_v
			i += 1
			
			# print(delta_l,delta_v)
		except:
			# print(doi)
			pass
	print(i)
	return X,Y




# plt.scatter(X1,Y1,alpha=0.05,s=2)
# plt.xlabel("$\\Delta$l")
# plt.ylabel("$\\Delta$v")
# plt.yscale('log')
# plt.savefig('test_vectors_orig.png')

import seaborn as sns

def seaborn_plot(X2,Y2,filename):
	df = pd.DataFrame({"$\\Delta$l":X2,"$\\Delta$v":Y2,})
	g = sns.jointplot(x="$\\Delta$l",
	              y="$\\Delta$v",
	              data = df ,
	              kind="scatter",
	              joint_kws =dict(alpha=0.1)
	              )
	g.ax_joint.set_yscale('log')
	plt.ylim([10,1000000])
	plt.xlim([0,12])
	g.ax_joint.set_xlabel("$\\Delta$l",fontsize=18)
	g.ax_joint.set_ylabel("$\\Delta$v",fontsize=18)
	plt.savefig(filename)


file = 'segm/mse_curves_syn_original.txt'
mse = open(file,'r').read().split('\n')[:-1]
i = 0
bla = []
for sample in mse:
	a = float(sample)
	bla.append(a)
	i += 1
print('numero de mse',i)

data_dict_ori,data_segm_dict_ori = get_orig_data()
data_dict_syn,data_segm_dict_syn = get_syn_data(bla)



for k in [5]:
	X1,Y1 = get_X_Y(data_dict_ori,data_segm_dict_ori,k)

	X2,Y2 = get_X_Y(data_dict_syn,data_segm_dict_syn,k)

	filename = 'vectors_diff_seaborn_orig_%d.png'%k
	seaborn_plot(X1,Y1,filename)

	filename = 'vectors_diff_seaborn_syn_%d.png'%k
	seaborn_plot(X2,Y2,filename)
