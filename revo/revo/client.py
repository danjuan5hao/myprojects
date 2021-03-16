# -*- coding: utf-8 -*-
import requests

if __name__ == "__main__":
    
    url_params = {"user_id": 1024, "utterance": "我喜欢自然语言处理"}

    rst = requests.get("http://127.0.0.1:5000/", params = url_params)
    print(rst.text) 