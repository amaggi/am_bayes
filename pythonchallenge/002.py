import urllib2
import numpy as np

url="http://www.pythonchallenge.com/pc/def/ocr.html"
page =urllib2.urlopen(url)
data=page.read()
print data

start_index = data.rfind('<!--')+4
end_index = data.rfind('-->')
unique_chars = np.unique(data[start_index:end_index])

char_counts = []
chars = []
for i in xrange(len(unique_chars)):
    c = unique_chars[i]
    if not c.isspace():
        chars.append(c)
        char_counts.append(data[start_index:end_index].count(c))

s = ""
for i in xrange(len(chars)):
    if char_counts[i]==1:
        s = s+chars[i]
print s
