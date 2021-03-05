if __name__ == '__main__':
    filename = 'data\\plos_one_2019.txt'

    content = open(filename, 'r').read().split('\n')

    valid = []
    for i, line in enumerate(content):
        if i % 3 == 0:
            valid.append(line)

    dois = open('dois.txt', 'w')
    for v in valid:
        dois.write(v + '\n')

    dois.close()
