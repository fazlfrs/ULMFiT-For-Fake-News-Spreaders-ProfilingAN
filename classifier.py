from fastai import *
from fastai.text import *
import pandas as pd
#--------------------------------------------------------------------

PATH=Path(os.getcwd())#/home/fazl/Desktop/fake_gen/code
csv_path = PATH/'csv'
mdl_path = csv_path/'models'


#---------------------------------------------------------------------
en_df=pd.read_csv(csv_path/'en.csv')
es_df=pd.read_csv(csv_path/'es.csv')

new_en_df=pd.concat([en_df['label'],en_df['tweets']],axis=1)
new_en_df.to_csv(csv_path/'new_en.csv',index=False)

new_es_df=pd.concat([es_df['label'],es_df['tweets']],axis=1)
new_es_df.to_csv(csv_path/'new_es.csv',index=False, header=False)

#-------------------------------------------------
print('loading Data language model---en')
#English
tokenizer = Tokenizer(lang='en', n_cpus=6)
en_data_lm = (TextList.from_csv(csv_path, csv_name='enfulltrain.csv', cols=0, processor=[TokenizeProcessor(tokenizer=tokenizer), NumericalizeProcessor(max_vocab=60000)])
           #Inputs: all the text files in path
            .split_from_df(col=1)
           #We may have other temp folders that contain text files so we only keep what's in train and test
            .label_for_lm()           
           #We want to do a language model so we label accordingly
            .databunch(bs=32))
print('loading Data language model---es')
#Spanish
tokenizer = Tokenizer(lang='es', n_cpus=6)
es_data_lm = (TextList.from_csv(csv_path, csv_name='esfulltrain.csv', cols=0, processor=[TokenizeProcessor(tokenizer=tokenizer), NumericalizeProcessor(max_vocab=60000)])
           #Inputs: all the text files in path
            .split_from_df(col=1)
           #We may have other temp folders that contain text files so we only keep what's in train and test
            .label_for_lm()           
           #We want to do a language model so we label accordingly
            .databunch(bs=32))

#-------------------------------------------
print('loading Data class model---en')
en_data_clas = TextClasDataBunch.from_csv(csv_path, vocab=en_data_lm.train_ds.vocab, bs=64, csv_name='new_en.csv', min_freq=1)
print('loading Data class model---es')
es_data_clas = TextClasDataBunch.from_csv(csv_path, vocab=es_data_lm.train_ds.vocab, bs=64, csv_name='new_es.csv', min_freq=1)

print('loading fine tuned encoder---en')
en_learn = text_classifier_learner(en_data_clas, AWD_LSTM, drop_mult=0.3,pretrained = False)
en_learn.load_encoder(mdl_path/'en_ft_enc')
en_learn.freeze()
print('loading fine tuned encoder---es')
es_learn = text_classifier_learner(es_data_clas, AWD_LSTM, drop_mult=0.3,pretrained = False)
es_learn.load_encoder(mdl_path/'es_ft_enc')
es_learn.freeze()
#-----------------------------------------------------
print('learning classifier ...')

en_learn.lr_find(start_lr=1e-8,end_lr=1e2)
#en_learn.recorder.plot()
for i in range(3):
	en_learn.freeze_to(-2)
	en_learn.fit_one_cycle(1, slice(1e-4,1e-2), moms=(0.8,0.7))

	en_learn.freeze_to(-3)
	en_learn.fit_one_cycle(1, slice(1e-5,5e-3), moms=(0.8,0.7))
	en_learn.unfreeze()
	en_learn.fit_one_cycle(4, slice(1e-5,1e-3), moms=(0.8,0.7))

	en_learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
#-------------------------------------------------------------------------
es_learn.lr_find()
#es_learn.recorder.plot(skip_start=0)
for i in range(3):
	es_learn.freeze_to(-2)
	es_learn.fit_one_cycle(1, slice(1e-4,1e-2), moms=(0.8,0.7))

	es_learn.freeze_to(-3)
	es_learn.fit_one_cycle(1, slice(1e-5,5e-3), moms=(0.8,0.7))
	es_learn.unfreeze()
	es_learn.fit_one_cycle(4, slice(1e-5,1e-3), moms=(0.8,0.7))

	es_learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))

#-----------------------------------------------------------------------
out_path=PATH/'output'
out_path.mkdir(exist_ok=True)

en_path=out_path/'en'
en_path.mkdir(exist_ok=True)

es_path=out_path/'es'
es_path.mkdir(exist_ok=True)
#ENGLISH
print('prediction---en')

test=en_df
test = test.drop(['label'], axis=1)

test['label'] = test['tweets'].apply(lambda row: str(en_learn.predict(row)[0]))

print(test['label'].value_counts())

test.to_csv(PATH/'enpredicted.txt', sep='\t', index=True, header=['author','tweets','label'], index_label='ID')
engold=en_df
engold.to_csv(PATH/'engold.txt', sep='\t', index=True, header=['author','tweets','label'], index_label='ID')
#------

for t in range(len(test)):
  
  id_=test['author'][t]
  lang='en'
  if test['label'][t]=='1':
    type_='fake'
  else:
    type_='genuine'
  res='<author id='+id_+'  lang='+lang+' type='+type_+' />'
  name=id_+".xml"
  with open(en_path/name, "w") as f:
    f.write(res)
#----

#Spanish
print('prediction---es')
test=es_df
test = test.drop(['label'], axis=1)
test['label'] = test['tweets'].apply(lambda row: str(es_learn.predict(row)[0]))
print(test['label'].value_counts())
test.to_csv(PATH/'espredicted.txt', sep='\t', index=True, header=['author','tweets','label'], index_label='ID')
esgold=en_df
esgold.to_csv(PATH/'esgold.txt', sep='\t', index=True, header=['author','tweets','label'], index_label='ID')
#--------------------------------------------------------------------------------------------------------------------

#------

for t in range(len(test)):
  
  id_=test['author'][t]
  lang='es'
  if test['label'][t]=='1':
    type_='fake'
  else:
    type_='genuine'
  res='<author id='+id_+'  lang='+lang+' type='+type_+' />'
  name=id_+".xml"
  with open(es_path/name, "w") as f:
    f.write(res)
#----

