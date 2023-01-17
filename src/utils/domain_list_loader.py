import re
from urllib.parse import urlparse
import pandas as pd

source_frame = pd.read_pickle('data/3k_spreaders_full.gzip', compression='gzip')
class_filter = (source_frame['fake_news_spreader'] == 0)
source_frame = source_frame[class_filter]

domain_map = {}

def get_links_from_post(post):
    links = re.findall(
        r'(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))',
        post)
    return [link[0] for link in links]

def get_domains(post):
    return [urlparse(link).netloc for link in get_links_from_post(post)]

for index, row in source_frame.iterrows():
    for doc in row['documents']:
        for domain in get_domains(doc[1]):
            if domain in domain_map.keys():
                domain_map[domain] += 1
            else:
                domain_map[domain] = 1

res = {k: v for k, v in sorted(domain_map.items(), key=lambda item: item[1]) if v > 50}

[print(link.replace('www.', '')) for link in res]