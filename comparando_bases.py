filename1 = 'data/plos_one_2019_breakpoints_k4_original1_data.txt'
filename2 = 'r_code/teste.txt'

file1 = open(filename1,'r').readlines()
file2 = open(filename2,'r').readlines()

N = len(file1)
set1 = set()
set2 = set()
for i in range(0,N,4):
	if file1[i] != file2[i]:
		print(file1,file2)