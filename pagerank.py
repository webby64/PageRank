import os
import random
import re
import sys
import collections

import numpy

DAMPING = 0.85
SAMPLES = 10000


def main():
    print("main")
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    ret = collections.defaultdict(float)
    if len(corpus[page]):
        for neigh in corpus[page]:
            ret[neigh] += damping_factor/float(len(corpus[page]))

        for p in corpus:
            ret[p] += (1-damping_factor)/float(len(corpus))
    else:
        for p in corpus:
            ret[p] = 1/float(len(corpus))
    return ret


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    retdict = collections.defaultdict(float)
    onepage = [random.choice(list(corpus.keys()))]
    for i in range(n):
        retdict[onepage[0]] += 1.0/float(n)
        therelation = transition_model(corpus, onepage[0], damping_factor)
        listOfpages = list(therelation.keys())
        probabilities = [therelation[i] for i in listOfpages]

        onepage = numpy.random.choice(listOfpages,1,p=probabilities)

    return retdict



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageRank = list(corpus.keys())
    n = len(corpus)
    pageRank = {pageRank[i]: 1.0/float(len(pageRank)) for i in range(len(pageRank))}
    numLinks = collections.defaultdict(int)
    count = 0
    for val in corpus.values():
        if not val:
            count +=1
        for i in val:
            numLinks[i]+=1

    while True:
        thebreak = 0
        for page in pageRank:

            if corpus[page]:
                temp = (1-damping_factor)/float(n) + damping_factor*sum([pageRank[i]/float(numLinks[i]) for i in corpus[page] if numLinks[i]!=0])
            else:

                temp = ((1-damping_factor)/float(n)) + \
                       damping_factor*(pageRank[page]/numLinks[page] if numLinks[page] != 0 else 0)

            if abs(temp - pageRank[page]) <= 0.0001:
                thebreak +=1
            pageRank[page] = temp
        if thebreak == n:
            break



    return pageRank


if __name__ == "__main__":
    main()
