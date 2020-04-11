import matplotlib.pyplot as plt

f = 'valid_example.txt'

content = open(f,'r').read()
content = content.split('\n\n')

seq = [float(d) for d in content[0].split()]
yy = [float(d) for d in content[1].split()]
xx = [float(d) for d in content[2].split()]
breaks = [float(d) for d in content[3].split()]


plt.figure(figsize=(5,5))

plt.plot(xx,seq,'-',c='tab:red',alpha=0.6)

for x in breaks:
	plt.axvline(x,c='tab:gray',alpha=0.6)

plt.scatter(xx,yy,c='tab:green',s=10)

plt.xlabel('time',fontsize=16)
plt.ylabel('views',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('valid_example.pdf')


#deu ruim [1] "10.1371/journal.pone.0002051"
# [1] "10.1371/journal.pone.0000341"