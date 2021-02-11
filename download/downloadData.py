import WOSGetter
import multiprocessing
import json
import sys
import traceback

import urllib
import os.path
import urllib.request
from urllib.parse import urlencode
from http.cookiejar import CookieJar, MozillaCookieJar
from multiprocessing.dummy import Pool as ThreadPool

requestTrials = 3
connectionTimeout = 60
processTrials = 3
errorVerbose = True


def dorequest(url, cj=None, data=None, timeout=10, encoding='UTF-8'):
    data = urlencode(data).encode(encoding) if data else None

    request = urllib.request.Request(url)
    request.add_header('User-Agent', 'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)')
    f = urllib.request.urlopen(request, data, timeout=timeout)
    return f.read()


def getURL(url):
    k = 0;
    theContents = None;
    while ((theContents is None) and k < requestTrials):
        k += 1;
        try:
            theContents = dorequest(url, timeout=connectionTimeout).decode("utf-8-sig")
        except urllib.error.URLError as e:
            theContents = None;
            if (errorVerbose):
                sys.stdout.write("\n##" + ("=" * 40) + "\n");
                sys.stdout.write("##%r (%s) url = %s\n" % (e, str(sys.exc_info()[0]), url));
                traceback.print_exc(file=sys.stdout);
                sys.stdout.write("\n##" + ("=" * 40) + "\n");
                sys.stdout.flush();
        except:
            theContents = None;
            if (errorVerbose):
                sys.stdout.write("\n##" + ("=" * 40) + "\n");
                sys.stdout.write(
                    "##A general exception occurred (%s). Could not process URL: %s\n" % (str(sys.exc_info()[0]), url));
                traceback.print_exc(file=sys.stdout);
                sys.stdout.write("\n##" + ("=" * 40) + "\n");
                sys.stdout.flush();
    if (theContents is None):
        sys.stdout.write("\n!!WARNING: m trials reached. Could not process URL: %s\n" % url);
        sys.stdout.flush();
    return theContents;


def processInput(doi):
    # if(counter.value%10==0):
    # 	print("Tokenizing: %d/%d             "%(counter.value,counterMaximum),end="\r");
    # with counter_lock:
    # 	counter.value += 1;
    # doi = DOIURL;#'10.1371/journal.pone.0073791';
    if (len(doi) > 0):
        data = getURL('http://alm.plos.org/api/v5/articles?api_key=3pezRBRXdyzYW6ztfwft&ids=' + urllib.parse.quote(doi,
                                                                                                                   safe='') + '&info=detail');
        return data;
    else:
        return None;


def save(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load(filename):
    data = None
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def get_papers(i_DOIs):

    doi_json = dict()

    data = processInput(i_DOIs)
    json_data = None
    selected = dict()

    try:
        json_data = json.loads(data)

        current = json_data['data'][0]
        selected[current['id']] = {}

        for a in current['sources']:

            if a['name'] == 'scopus':
                selected[current['id']] = a['events']['citedby-count']
        # if a['name'] == 'counter':
        # 	selected[current['id']]['views'] = a['events']
        # if a['name'] == 'twitter':
        # 	selected[current['id']]['twitter'] = {'freq':a['by_month'],'elements':a['events']}

    except:
        json_data = None

    if json_data is not None:
        # doi_json[doi] = selected
        return selected

    # save(doi_json, 'papers_time_series/papers_time_series' + str(head) + '_scopus_count_2.json')


if __name__ == '__main__':
    wosArticles = WOSGetter.GZJSONLoad("wosPlosOne2016_citations.json.gz")
    # wosNetwork = xnet.xnet2igraph("wosPlosOne2016_cocitation.xnet")

    cj = MozillaCookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    urllib.request.install_opener(opener)

    cookie_file = os.path.abspath('./cookies.txt')



    # what are your inputs, and what operation do you want to
    # perform on each input. For example...
    DOIs = [article["DI"] for article in wosArticles];
    del wosArticles

    n = 20000
    # chuncks = [(i,DOIs[i:i + n]) for i in range(0, len(DOIs), n)]

    pool = multiprocessing.Pool(6)
    results = pool.map(get_papers, DOIs, chunksize=n)
    save(results, 'papers_time_series/papers_time_series_scopus_count_2.json')