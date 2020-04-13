import matplotlib.pyplot as plt

def plt_example(f_input,f_output):
	

	content = open(f_input,'r').read()
	content = content.split('\n\n')

	seq = [float(d) for d in content[0].split()]
	yy = [float(d) for d in content[1].split()]
	xx = [float(d) for d in content[2].split()]
	breaks = [float(d) for d in content[3].split()]


	plt.figure(figsize=(5,5))

	plt.plot(xx,seq,'-',c='tab:red',alpha=1,linewidth=3)

	for x in breaks:
		plt.axvline(x,c='tab:gray',alpha=0.6)

	plt.scatter(xx,yy,c='tab:green',s=40,alpha=0.8)

	plt.xlabel('time',fontsize=16)
	plt.ylabel('views',fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.tight_layout()
	plt.savefig(f_output)

plt_example('valid_example.txt','invalid_example.pdf')
plt_example('invalid_example.txt','invalid_example.pdf')

# "10.1371/journal.pone.0014108" [1] "0.000201"
