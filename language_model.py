from fastai import *
from fastai.text import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import preprocessor as p
import pandas as pd

#--------------------------------------------------------------------
file_path=Path('/content/')
PATH=Path('/content/drive/My Drive/Pan_test/code')#gh/home/fazl/Desktop/fake_gen/code
csv_path = PATH/'csv'
csv_path.mkdir(exist_ok=True)
mdl_path = csv_path/'models'
mdl_path.mkdir(exist_ok=True)

#---------------------------------------------------------------------

#English
enfake=open(file_path/'prepared_dataset/en/train_fake.txt').read().split('\n')
enFid=open(file_path/'prepared_dataset/en/fakeid.txt').read().split('\n')
engen=open(file_path/'prepared_dataset/en/train_genuine.txt').read().split('\n')
enGid=open(file_path/'prepared_dataset/en/genid.txt').read().split('\n')

#Spanish
esfake=open(file_path/'prepared_dataset/es/train_fake.txt').read().split('\n')
esFid=open(file_path/'prepared_dataset/es/fakeid.txt').read().split('\n')
esgen=open(file_path/'prepared_dataset/es/train_genuine.txt').read().split('\n')
esGid=open(file_path/'prepared_dataset/es/genid.txt').read().split('\n')

#English
label=[]
tweets=[]
author=[]
for i in range(len(enfake)):
    label.append('fake')
    tweets.append(enfake[i])
    author.append(enFid[i])
for j in range(len(engen)):
    label.append('genuine')
    tweets.append(engen[j])
    author.append(enGid[j])
endict={'author':author,'tweets':tweets,'label':label}

#Spanish
label=[]
tweets=[]
author=[]
for i in range(len(esfake)):
    label.append('fake')
    tweets.append(esfake[i])
    author.append(esFid[i])
for j in range(len(esgen)):
    label.append('genuine')
    tweets.append(esgen[j])
    author.append(esGid[j])
esdict={'author':author,'tweets':tweets,'label':label}

#--------------------------------------------------------------------

def preprocessing1(tweets, _stopwords):
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.NUMBER, p.OPT.RESERVED)
    lemmatizer = WordNetLemmatizer()
    cleaned_tweets = p.clean(tweets)
    word_list = cleaned_tweets.split()    
    mod_tweet = []
    for word in word_list:
        if word.lower() in _stopwords:
            continue
        mod_tweet.append(lemmatizer.lemmatize(word.lower()))      
    return ' '.join(mod_tweet)

print('preprocessing ....')
en_df=pd.DataFrame(endict)
es_df=pd.DataFrame(esdict)
en_df['tweets']=en_df['tweets'].apply(lambda x:preprocessing1(x,set(stopwords.words('english'))))
es_df['tweets']=es_df['tweets'].apply(lambda x:preprocessing1(x,set(stopwords.words('spanish'))))
en_df.to_csv(csv_path/'en.csv', index=None)
es_df.to_csv(csv_path/'es.csv', index=None)

#----------------------------------------------------------------
es_lmText=pd.read_csv(csv_path/'es_lmText.csv')
en_lmText=pd.read_csv(csv_path/'en_lmText.csv')

en_df_train=en_lmText[:800000]
es_df_train=es_lmText[:800000]

en_df_valid=en_lmText[800000:1000000]
es_df_valid=es_lmText[800000:1000000]


en_df_train['is_valid'] = False
en_df_valid['is_valid'] = True

es_df_train['is_valid'] = False
es_df_valid['is_valid'] = True

en_regroup = pd.concat([en_df_train, en_df_valid])
es_regroup = pd.concat([es_df_train, es_df_valid])

en_regroup.to_csv(csv_path/'enfulltrain.csv', header=None, index=None)
es_regroup.to_csv(csv_path/'esfulltrain.csv', header=None, index=None)
#-------------------------------------------------------------------------------------

#*****LM Training*********
print('LM Training ...')
#English
tokenizer = Tokenizer(lang='en', n_cpus=6)
en_data_lm = (TextList.from_csv(csv_path, csv_name='enfulltrain.csv', cols=0, processor=[TokenizeProcessor(tokenizer=tokenizer), NumericalizeProcessor(max_vocab=60000)])
           #Inputs: all the text files in path
            .split_from_df(col=1)
           #We may have other temp folders that contain text files so we only keep what's in train and test
            .label_for_lm()           
           #We want to do a language model so we label accordingly
            .databunch(bs=32))

#Spanish
tokenizer = Tokenizer(lang='es', n_cpus=6)
es_data_lm = (TextList.from_csv(csv_path, csv_name='esfulltrain.csv', cols=0, processor=[TokenizeProcessor(tokenizer=tokenizer), NumericalizeProcessor(max_vocab=60000)])
           #Inputs: all the text files in path
            .split_from_df(col=1)
           #We may have other temp folders that contain text files so we only keep what's in train and test
            .label_for_lm()           
           #We want to do a language model so we label accordingly
            .databunch(bs=32))
print('language model learner --- en')
en_learn = language_model_learner(en_data_lm, AWD_LSTM, drop_mult=0.5, pretrained=False)
print('language model learner --- es')
es_learn = language_model_learner(es_data_lm, AWD_LSTM, drop_mult=0.5, pretrained=False)

lr=1e-5
print('language model learner fit --- en')
en_learn.fit_one_cycle(5, lr, moms=(0.8,0.7))
print('language model learner --- es')
es_learn.fit_one_cycle(5, lr, moms=(0.8,0.7))

#--------------------------------------------------------
lang = 'en'
lm_fns = [f'{lang}_wt', f'{lang}_wt_vocab']
en_learn.save(mdl_path/lm_fns[0], with_opt=False)
en_learn.data.vocab.save(mdl_path/(lm_fns[1] + '.pkl'))


lang = 'es'
lm_fns = [f'{lang}_wt', f'{lang}_wt_vocab']
es_learn.save(mdl_path/lm_fns[0], with_opt=False)
es_learn.data.vocab.save(mdl_path/(lm_fns[1] + '.pkl'))

#---------------------------------------------------------------

# **************fine tuning********************
weights_pretrained = 'en_wt'
itos_pretrained = 'en_wt_vocab'
en_pretained_data = (weights_pretrained, itos_pretrained)

weights_pretrained = 'es_wt'
itos_pretrained = 'es_wt_vocab'
es_pretained_data = (weights_pretrained, itos_pretrained)

en_learn = language_model_learner(en_data_lm, AWD_LSTM, pretrained_fnames=en_pretained_data, drop_mult=0)
en_learn.freeze()

es_learn = language_model_learner(es_data_lm, AWD_LSTM, pretrained_fnames=es_pretained_data, drop_mult=0)
es_learn.freeze()

en_learn.lr_find()
en_learn.recorder.plot(skip_start=0)

es_learn.lr_find()
es_learn.recorder.plot(skip_start=0)

en_learn.fit_one_cycle(1, 1e-2)
en_learn.save(mdl_path/'en_head_pretrained')
en_learn.unfreeze()
en_learn.fit_one_cycle(2, 1e-3, moms=(0.8,0.7))
en_learn.save(mdl_path/'en_lm_fine_tuned')

es_learn.fit_one_cycle(1, 1e-2)
es_learn.save(mdl_path/'es_head_pretrained')
es_learn.unfreeze()
es_learn.fit_one_cycle(2, 1e-3, moms=(0.8,0.7))
es_learn.save(mdl_path/'es_lm_fine_tuned')

# Save the fine-tuned encoder
en_learn.save_encoder(mdl_path/'en_ft_enc')
es_learn.save_encoder(mdl_path/'es_ft_enc')
print('Done!')
