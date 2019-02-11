"""Script to download and clean Shakespeare dataset."""

import requests
import re
import io
import zipfile

# url = "http://www.gutenberg.org/files/100/100-0.txt"
url = "https://storage.googleapis.com/" \
      "download.tensorflow.org/data/shakespeare.txt"
text = requests.get(url).text

# removed speaker names and newline spacing
processed_text = re.sub(("^.*?\\n|\\n\\n.*?\\n|\\n|[A-Z]+:"), " ", text)

# saves to zip
mf = io.BytesIO()
with zipfile.ZipFile(mf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.writestr('clean_shakespeare.txt', str.encode(processed_text, 'utf-8'))

with open("../data/clean_shakespeare.zip", "wb") as f:  # use `wb` mode
    f.write(mf.getvalue())
