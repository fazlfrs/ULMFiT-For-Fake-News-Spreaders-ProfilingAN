from fastai import *
from fastai.text import *
import pandas as pd
import csv
import glob
import os
bs=128
data_path =Path('/content/drive/My Drive/Pan_test/collected_data')
data_path.mkdir(exist_ok=True)
csv_path=Path('/content/drive/My Drive/Pan_test/code/csv')
csv_path.mkdir(exist_ok=True)

lang = 'es'
name = f'{lang}wiki'
path = data_path/name
path.mkdir(exist_ok=True, parents=True)


def get_wiki(path,lang):
    name = f'{lang}wiki'
    if (path/name).exists():
        print(f"{path/name} already exists; not downloading")
        return

    xml_fn = f"{lang}wiki-latest-pages-articles.xml"
    zip_fn = f"{xml_fn}.bz2"

    if not (path/xml_fn).exists():
        print("downloading...")
        download_url(f'https://dumps.wikimedia.org/{name}/latest/{zip_fn}', path/zip_fn)
        print("unzipping...")
        bunzip(path/zip_fn)

    with working_directory(path):
        if not (path/'wikiextractor').exists(): os.system('git clone https://github.com/attardi/wikiextractor.git')
        print("extracting...")
        os.system("python wikiextractor/WikiExtractor.py --processes 4 --no_templates " +
            f"--min_text_length 1800 --filter_disambig_pages --log_file log -b 100G -q {xml_fn}")
    shutil.move(str(path/'text/AA/wiki_00'), str(path/name))
    shutil.rmtree(path/'text')
    
get_wiki(path,lang)

def split_wiki(path,lang):
    dest = path/'docs'
    name = f'{lang}wiki'
    if dest.exists():
        print(f"{dest} already exists; not splitting")
        return dest

    dest.mkdir(exist_ok=True, parents=True)
    title_re = re.compile(rf'<doc id="\d+" url="https://{lang}.wikipedia.org/wiki\?curid=\d+" title="([^"]+)">')
    lines = (path/name).open()
    f=None

    for i,l in enumerate(lines):
        
        if i%100000 == 0: print(i)
        if l.startswith('<doc id="'):
            title = title_re.findall(l)[0].replace('/','_')
            if len(title)>150: continue
            if f: f.close()
            f = (dest/f'{title}.txt').open('w')
        else: f.write(l)
    f.close()
    return dest
    
dest=split_wiki(path,lang)


directory = dest


txt_files = os.path.join(directory, '*.txt')
mylist=[]
for txt_file in glob.glob(txt_files):
    txt=open(txt_file,'r').read().replace('\n','').replace('</doc>','')
    mylist.append(txt)

dict1={'content':mylist}
df=pd.DataFrame(dict1)
df.to_csv(csv_path/'es_lmText.csv',index=False)
print('Spanish wiki downloaded and saved!')


#*******************************

lang = 'en'
name = f'{lang}wiki'
path = data_path/name
path.mkdir(exist_ok=True, parents=True)


def get_wiki(path,lang):
    name = f'{lang}wiki'
    if (path/name).exists():
        print(f"{path/name} already exists; not downloading")
        return

    xml_fn = f"{lang}wiki-latest-pages-articles.xml"
    zip_fn = f"{xml_fn}.bz2"

    if not (path/xml_fn).exists():
        print("downloading...")
        download_url(f'https://dumps.wikimedia.org/{name}/latest/{zip_fn}', path/zip_fn)
        print("unzipping...")
        bunzip(path/zip_fn)

    with working_directory(path):
        if not (path/'wikiextractor').exists(): os.system('git clone https://github.com/attardi/wikiextractor.git')
        print("extracting...")
        os.system("python wikiextractor/WikiExtractor.py --processes 4 --no_templates " +
            f"--min_text_length 1800 --filter_disambig_pages --log_file log -b 100G -q {xml_fn}")
    shutil.move(str(path/'text/AA/wiki_00'), str(path/name))
    shutil.rmtree(path/'text')
    
get_wiki(path,lang)

def split_wiki(path,lang):
    dest = path/'docs'
    name = f'{lang}wiki'
    if dest.exists():
        print(f"{dest} already exists; not splitting")
        return dest

    dest.mkdir(exist_ok=True, parents=True)
    title_re = re.compile(rf'<doc id="\d+" url="https://{lang}.wikipedia.org/wiki\?curid=\d+" title="([^"]+)">')
    lines = (path/name).open()
    f=None

    for i,l in enumerate(lines):
        
        if i%100000 == 0: print(i)
        if l.startswith('<doc id="'):
            title = title_re.findall(l)[0].replace('/','_')
            if len(title)>150: continue
            if f: f.close()
            f = (dest/f'{title}.txt').open('w')
        else: f.write(l)
    f.close()
    return dest
    
dest=split_wiki(path,lang)


directory = dest


txt_files = os.path.join(directory, '*.txt')
mylist=[]
for txt_file in glob.glob(txt_files):
    txt=open(txt_file,'r').read().replace('\n','').replace('</doc>','')
    mylist.append(txt)

dict1={'content':mylist}
df=pd.DataFrame(dict1)
df.to_csv(csv_path/'en_lmText.csv',index=False)
print('English wiki downloaded and saved!')

df.to_csv(csv_path/'en_lmText.csv',index=False)
print('English wiki downloaded and saved!')

