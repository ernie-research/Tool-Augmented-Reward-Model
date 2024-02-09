import os
import random
import requests
import hashlib

class BaiduTranslator:
    def __init__(self) -> None:
        self.BAIDU_TRANSLATE_KEY  = ""  # INPUT your translate key
        self.BAIDU_SECRET_KEY = "" # INPUT your secret key
        self.URL = 'https://fanyi-api.baidu.com/api/trans/vip/translate'

    def get_translation(self, text:str, source_lang: str='auto', tgt_lang:str='en') -> str:
        salt = random.randint(32768, 65536)
        sign = self.BAIDU_TRANSLATE_KEY + text + str(salt) + self.BAIDU_SECRET_KEY
        md = hashlib.md5()
        md.update(sign.encode(encoding='utf-8'))
        sign =md.hexdigest()
        header = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {
            "appid": self.BAIDU_TRANSLATE_KEY,
            "q": text,
            "from": source_lang,
            "to": tgt_lang,
            "salt": salt,
            "sign": sign
        }
        response = requests.post(self.URL, params=data, headers=header)
        text = response.json()
        results = text['trans_result'][0]['dst']
        return results
