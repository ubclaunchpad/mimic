"""Script to download dataset for machine translation."""

import requests
import os
import io
import zipfile

url_eng_train = "https://nlp.stanford.edu/\
                projects/nmt/data/iwslt15.en-vi/train.en"
url_viet_train = "https://nlp.stanford.edu/\
                 projects/nmt/data/iwslt15.en-vi/train.vi"
english_train = requests.get(url_eng_train).text
viet_train = requests.get(url_viet_train).text

url_eng_test = "https://nlp.stanford.edu/\
               projects/nmt/data/iwslt15.en-vi/tst2013.en"
url_viet_test = "https://nlp.stanford.edu/\
               projects/nmt/data/iwslt15.en-vi/tst2013.vi"
english_test = requests.get(url_eng_test).text
viet_test = requests.get(url_viet_test).text

file_list = {"eng_train": english_train,
             "viet_train": viet_train,
             "eng_test": english_test,
             "viet_test": viet_test}

os.makedirs("../data/english-vietnamese")

for file in file_list:
    mf = io.BytesIO()

    with zipfile.ZipFile(mf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(file + '.txt', str.encode(file_list[file], 'utf-8'))

    with open("../data/english-vietnamese/" + file + ".zip", "wb") as f:
        f.write(mf.getvalue())
