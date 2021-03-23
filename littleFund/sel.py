# -*- coding: utf-8 -*-

from selenium import webdriver 
from selenium.webdriver.common.by import By 
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.support.wait import WebDriverWait 

import requests 
from bs4 import BeautifulSoup

browser = webdriver.Chrome()
url = "http://fund.eastmoney.com/manager/default.html#dt14;mcreturnjson;ftall;pn50;pi1;scabbname;stasc"

browser.get(url)
print(browser.page_source)
# rst = requests.get(url)
# print(rst.status_code)
# # print(rst.content)

# soup = BeautifulSoup(rst.content, "lxml")
# print(soup.title.string)

# rst = soup.find_all("div", "datatable")
# print(len(rst))
# tbodys = rst[0].find_all("tbody")
# print(len(tbodys))
# trs = tbodys[0]
# print(type(trs))
# rst2 = trs.find_all("tr")

