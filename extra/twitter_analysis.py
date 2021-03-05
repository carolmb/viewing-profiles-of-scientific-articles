import glob
import json

if __name__ == '__main__':
    # files = glob.glob('others\\papers_time_series\\papers_time_series*2.json')
    # twitter_dict = dict()
    # for file in files:
    #     content = open(file, 'r')
    #     content_json = json.load(content)
    #     for doi, series in content_json.items():
    #         for _, series1 in series.items():
    #             tweets = series1['twitter']['elements']
    #             twitter_data = []
    #             for tweet in tweets:
    #                 twitter_data.append(tweet['event_time'])
    #             twitter_dict[doi] = twitter_data
    #
    # with open('twitter_data.json', 'w') as outfile:
    #     json.dump(twitter_dict, outfile)
    data = open('twitter_data.json', 'r')
    twitter_data = json.load(data)
    count = 0
    total = 0
    for doi, tweets in twitter_data.items():
        total += 1
        if len(tweets) >= 1:
            count += 1
    print('count', count)
    print('total', total)