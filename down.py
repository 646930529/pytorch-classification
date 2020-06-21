import re
import requests
from urllib.parse import urlparse
import os
import time
import sys


savedir = 'cup2'
word = '水杯'
url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + word + '&pn='
maxdown = 1000
pagesize = 60


def mkh(url):
    res=urlparse(url)
    headers = {
        #'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        #'Accept-Encoding': 'gzip, deflate',
        #'Accept-Language': 'zh-CN,zh;q=0.9',
        #'Cache-Control': 'no-cache',
        #'Connection': 'keep-alive',
        'Host': res.netloc,
        #'Pragma': 'no-cache',
        #'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.106 Safari/537.36'
    }
    return headers


def Find(url):
    print('finding...')
    l = []
    t = 0
    while t < maxdown:
        print('Finding...', t)
        try:
            response = requests.get(url + str(t), timeout=7)
            if response.status_code != 200:
                print(response.status_code)
                t = t + pagesize
                continue
            pic_url = re.findall('"objURL":"(.*?)",', response.text, re.S)
            if len(pic_url):
                l.append(pic_url)
        except BaseException:
            print('time out t = ', t)
            t = t + pagesize
            continue
        except Exception as e:
            print(e)
        t = t + pagesize
    return l


l = Find(url)
print('finded page total = ', len(l))


print('downloading...')
download_index = 0
for i in l:
    for j in i:
        print('downloading...', download_index)
        try:
            if j is None:
                print(j)
                download_index += 1
                continue

            response = requests.get(j, headers=mkh(j), timeout=7)
            if response.status_code != 200:
                print(response.status_code, j)
                download_index += 1
                continue
            with open(savedir + '/' + word + '_' + str(download_index) + '.jpg', 'wb') as f:
                f.write(response.content)
        except BaseException:
            print('time out')
            download_index += 1
            continue
        except Exception as e:
            print(e)
        download_index += 1
        time.sleep(1)


