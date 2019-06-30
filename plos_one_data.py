import PlosOneData.WOSGetter as WOSGetter

completeTimeSeries = WOSGetter.GZJSONLoad("PlosOneData/plosone2016_hits.json.gz")
print(completeTimeSeries[0])