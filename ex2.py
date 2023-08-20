from collections import Counter
import math 
import random
import numpy as np

# generate data
def myData():
    data = []
    # add website to data
    data.append({'URL' : "fruits.com",
        'tokens' : ["a", "banana", "and", "watermelon", "apple", "pear"],
        'linksTo' : ["veg.com", "data.com", "betterfruits.com", "animals.com", "hungary.com", "hungry.com"]})

    data.append({'URL' : "veg.com",
        'tokens' : ["a", "banana", "and", "tomato", "apple", "potato", "pizza", "meat", "pizza"],
        'linksTo' : ["fruits.com", "betterfruits.com", "hungary.com", "pizza.com"]})

    data.append({'URL' : "pizza.com",
        'tokens' : ["pineapple", "and", "tuna", "should", "not", "be", "on", "a", "pizza", "not"],
        'linksTo' : ["fruits.com", "hungry.com", "veg.com", "data.com"]})

    data.append({'URL' : "coffee.com",
        'tokens' : ["tea", "maciato", "machine", "cofix", "tuna", "sugar", "onion", "cofix", "pear"], 
        'linksTo' : ["hungry.com"]})

    data.append({'URL' : "betterfruits.com", 
        'tokens' : ["a", "banana", "and", "watermelon", "strawberry", "pineapple", "tomato", "electric"], 
        'linksTo' : ["hungry.com", "veg.com"]})

    data.append({'URL' : "tools.com", 
        'tokens' : ["a", "hammer", "and", "an", "axe", "pear", "drill", "saw"], 
        'linksTo' : []})

    data.append({'URL' : "hungry.com", 
        'tokens' : ["hamburger", "banana", "and", "watermelon", "pizza", "dallas", "china", "chinese", "pasta"], 
        'linksTo' : ["fruits.com", "hungary.com"]})

    data.append({'URL' : "hungary.com", 
        'tokens' : ["budapest", "cube", "cold", "urban", "cold", "a", "goulash"], 
        'linksTo' : ["fruits.com", "hungry.com", "veg.com"]})

    data.append({'URL' : "data.com", 
        'tokens' : ["data", "base", "pineapple", "cold", "apple", "pear", "lamb"], 
        'linksTo' : ["fruits.com", "hungry.com", "veg.com", "tools.com"]})

    data.append({'URL' : "animals.com", 
        'tokens' : ["a", "banana", "and", "dragon", "lion", "lamb", "turkey", "pikachu", "pikachu", "pikachu", "dog", "cat"], 
        'linksTo' : ["fruits.com", "hungary.com", "veg.com"]})

    return data

# generate search string
def mySearchString():
    return ["I", "like", "strawberry", "pineapple", "and", "tomato", "on", "my", "electric", "pizza"]

# function for sorting
def sort_help(var):
    return var[-1]

# calculate inverted index score
def invertedIndex(data, searchString):
    ans = {}
    counters = {}
    # keep number of mentions for each word in each document
    for d in data:
        counters.update({d['URL'] : Counter(d['tokens'])})
    docs = len(counters)
    # calculate score for each word in the search string
    for token in searchString:
        s = 0
        tfs = {}
        # calculate tf
        for c in counters:
            words = sum(counters[c].values())
            if words == 0:
                tf = 0
            else:
                tf = counters[c].get(token, 0) / words
            tfs.update({c : tf})
            if tf != 0:
                s += 1
        # calculate idf
        idf = 0
        if s != 0:
            idf = math.log(10, docs/s)
        # keep tfidf in list and sort
        l = []
        for tf in tfs.items():
            l.append([tf[0], tf[-1]*idf])
        l.sort(reverse=True, key=sort_help)
        # add to ans
        ans.update({token : l})
    return ans

# simulation of random pageRank process
def pageRankSimulation(data, numIter, beta):
    docs = len(data)
    # initial values
    r = np.ones(docs)
    i = 0
    url_to_index = {}
    pages = {}
    for p in data:
        pages.update({p['URL'] : p['linksTo']})
        url_to_index.update({p['URL'] : i})
        i += 1
    urls = list(pages.keys())
    p = random.choice(urls)
    visits = numIter
    while numIter > 0:
        r[url_to_index[p]] += 1
        if random.uniform(0, 1) > beta:
            # damp
            p = random.choice(urls)
        else:
            # choose from neighbors
            if len(pages[p]) == 0:
                p = random.choice(urls)
            # handle no neighbors (trap)
            else:
                p = random.choice(pages[p])
        numIter -= 1
    if visits != 0:        
        r = r * (1/visits)

    ans = []
    for u, i in url_to_index.items():
        ans.append([u, r[i]])
    ans.sort(reverse=True, key=sort_help)
    return ans

# scoring function
def score(tfIdf, pageRankValue):
    s = 0
    # set thresholds
    pagerank_t = 0.04
    tfidf_t = 0.1
    r = 0.05
    if tfIdf > tfidf_t:
        exp = 1+math.floor(tfIdf /  r)
        s += math.pow(2, exp)
    if pageRankValue > pagerank_t:
        v = 1+math.floor(pageRankValue /  r)
        s += math.pow(v, 3)
    return s

# get a value by url, perfoms the random access
def rand_access(list, url):
    for sublist in list:
        if sublist[0] == url:
            return sublist[-1]

# calculate the threshold with the given data
def calc_threshold(d):
    tfidf = 0
    page_rank = 0
    for k, v in d.items():
        # set page rank score
        if k == 0:
            page_rank = v
        # add to tfidf score
        else:
            tfidf += v
    # calculate
    return score(tfIdf=tfidf,pageRankValue=page_rank)

# print access message
def print_access(url, access_t, index_t):
    txt = access_t + " access to " + url + " at the " + index_t + " index."
    print(txt)

# get matching text
def get_index_t(index):
    if index == 0:
        return "PageRank"
    return "tfIdf"

# top1 algorithm
def top1(invertedIndex, pageRank):
    # group the data (a list of lists of lists)
    lists = [pageRank]
    for k, v in invertedIndex.items():
        lists.append(v)
    # init threshold
    threshold = float('inf')
    ans = []
    i = 0
    # starting list
    l = lists[i]
    j = 0
    # dictionary, mapping url -> score(url) , used in order to keep the combined score of each url
    scores = {}
    # dictionary, mapping int -> int, used in order to keep the current minimal value of each list
    values = {}
    while len(ans) < 1:
        url = l[j][0]
        s = l[j][-1]
        # page rank score
        s_pageRank = 0
        # tfidf score
        s_tfidf = 0
        print_access(url=url, index_t=get_index_t(i), access_t="direct")
        values.update({i : l[j][1]})
        # check if we can calculate the threshold (if we have enough data)
        if j > 0 or i == len(lists) -1:
            threshold = calc_threshold(values)

        # if we didn't calculate it already
        if url not in scores.keys():
            if i == 0:
                # set page rank value
                rand_lists = lists.copy()
                rand_lists.remove(l)
                s_pageRank = s
            else:
                # add s to tfidf score and get page rank by random access
                rand_lists = lists.copy()
                rand_lists.remove(l)
                rand_lists.remove(lists[0])
                print_access(url=url, index_t="pageRank", access_t="random")
                s_pageRank = rand_access(lists[0], url)
                s_tfidf += s
            
            for list in rand_lists:
                print_access(url=url, index_t="tfIdf", access_t="random")
                # get tfidf score by random access
                s_tfidf += rand_access(list, url)
            # save the score
            scores.update({url : score(pageRankValue=s_pageRank, tfIdf=s_tfidf)})

        # next index
        i_new = (i + 1) % len(lists)
        if i_new < i:
            j += 1
        i = i_new
        l = lists[i]

        # add to ans if the score is greater than the threshold
        for k, v in scores.items():
            if v > threshold:
                ans.append((k,v)) 
    # return the max
    ans.sort(reverse=True, key=sort_help)
    return ans[0]

def main():
    print(top1(invertedIndex(data=myData(), searchString=mySearchString()), pageRankSimulation(data=myData(), numIter=100000, beta=0.8)))

if __name__ == "__main__":
    main()

