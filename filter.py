
f_input = 'data/plos_one_total_breakpoints_k4it.max100stop.if.errorFALSE.txt'

inp = open(f_input,'r').read().split('\n')

f_output = f_input[:-4] + '_filtered.txt'

out = open(f_output,'w')

for i in range(0,len(inp)-3,3):
	if '-' in inp[i+1]:
		continue
	out.write(inp[i]+'\n'+inp[i+1]+'\n'+inp[i+2]+'\n')

out.close()
