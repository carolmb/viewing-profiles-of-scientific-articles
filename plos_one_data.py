import WOSGetter

output = open('data/plos_one_data_total.txt','w')

completeTimeSeries = WOSGetter.GZJSONLoad("plosone2016_hits.json.gz")
for series in completeTimeSeries:
    try:
        ss = series['total']
    except:
        print(series)
    xs,ys = [],[]
    for s in ss:
        date = s[0].split('/')
        date = float(date[0])/12 + int(date[1])
        xs.append(str(date))
        ys.append(str(s[1]))
    output.write(','.join(xs)+'\n')
    output.write(','.join(ys)+'\n')

output.close()

