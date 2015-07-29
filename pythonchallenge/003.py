import urllib2
import numpy as np

url="http://www.pythonchallenge.com/pc/def/equality.html"
page =urllib2.urlopen(url)
data=page.read()
print data

