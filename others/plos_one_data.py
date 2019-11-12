import WOSGetter

# output = open('plos_one_data_total.txt','w')

completeTimeSeries = WOSGetter.GZJSONLoad("plosone2016_hits.json.gz")
for series in completeTimeSeries:
    pause = False
    try:
        ss = series['total']
    except:
        ss = []
        pause = True
    
    xs,ys = [],[]
    t = 0
    for s in ss:
        date = s[0].split('/')
        date = float(date[0])/12 + int(date[1])
        xs.append(str(date))
        ys.append(str(t+s[1]))
        t += s[1]
    output.write(','.join(xs)+'\n')
    output.write(','.join(ys)+'\n')


# output.close()
