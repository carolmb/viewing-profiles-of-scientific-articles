
# f_input = 'data/plos_one_2019_breakpoints_k4_original1_data.txt'
# f_input = 'data/plos_one_2019_menor.txt'
# f_input = 'r_code/segmented_curves.txt'
f_input = 'r_code/segmented_curves_syn_data.txt'

inp = open(f_input,'r').read().split('\n')

f_output = f_input[:-4] + '_filtered.txt'

out = open(f_output,'w')

for i in range(0,len(inp)-3,4):
	if '-' in inp[i+1]:
		continue
	out.write(inp[i]+'\n'+inp[i+1]+'\n'+inp[i+2]+'\n'+inp[i+3]+'\n')

out.close()

