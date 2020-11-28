from fastai import *
from fastai.text import *
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

#--------------------------------------------------------------------

#PATH=Path('/content/drive/My Drive/Pan_test/code')#/home/fazl/Desktop/fake_gen/code
PATH=Path(os.getcwd())
csv_path = PATH/'csv'
mdl_path = csv_path/'models'

out_path=PATH/'output'
out_path.mkdir(exist_ok=True)

en_path=out_path/'en'
en_path.mkdir(exist_ok=True)

es_path=out_path/'es'
es_path.mkdir(exist_ok=True)

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
print('learning classifier --- en')
#en_learn.lr_find(start_lr=1e-8,end_lr=1e2)
#en_learn.recorder.plot()
for i in range(2):
	en_learn.freeze_to(-2)
	en_learn.fit_one_cycle(1, slice(1e-4,1e-2), moms=(0.8,0.7))

	en_learn.freeze_to(-3)
	en_learn.fit_one_cycle(1, slice(1e-5,5e-3), moms=(0.8,0.7))
	en_learn.unfreeze()
	en_learn.fit_one_cycle(4, slice(1e-5,1e-3), moms=(0.8,0.7))

	en_learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
#-------------------------------------------------------------------------
print('\nprediction for English')
en_test=en_df
#test=pd.read_csv('/content/drive/My Drive/en_test_set.csv')
en_test['pre'] = en_test['tweets'].apply(lambda row: str(en_learn.predict(row)[0]))
#----------------------------------------------------------------------------

print('learning classifier---es ...')

#es_learn.lr_find()
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
print('prediction for Spanish')
test=es_df
#test=pd.read_csv('/content/drive/My Drive/es_test_set.csv')
test['pre'] = test['tweets'].apply(lambda row: str(es_learn.predict(row)[0]))

#---------------------------------------------------------------------

print("-------------------- English --------------------")
cm = confusion_matrix(en_test['label'],en_test['pre'])

print ("Confusion Matrix:\n",cm)
score1 = accuracy_score(en_test['label'],en_test['pre'])
score2 = precision_score(en_test['label'],en_test['pre'],average='micro')
score3= recall_score(en_test['label'],en_test['pre'],average='micro')
score4=f1_score(en_test['label'],en_test['pre'],average='micro')
print("\n")
print("Accuracy is ",round(score1*100,2),"%")
print("Precision is ",round(score2,2))
print("Recall is ",round(score3,2))
print('f1 score is ',round(score4,2))
print('\n')
print(classification_report(en_test['label'],en_test['pre']))


print("-------------------- Spanish --------------------")
cm = confusion_matrix(test['label'],test['pre'])

print ("Confusion Matrix:\n",cm)
score1 = accuracy_score(test['label'],test['pre'])
score2 = precision_score(test['label'],test['pre'],average='micro')
score3= recall_score(test['label'],test['pre'],average='micro')
score4=f1_score(test['label'],test['pre'],average='micro')
print("\n")
print("Accuracy is ",round(score1*100,2),"%")
print("Precision is ",round(score2,2))
print("Recall is ",round(score3,2))
print('f1 score is ',round(score4,2))
print('\n')
print(classification_report(test['label'],test['pre']))

#----------------------------------------------------------------
print('storing output in ',str(out_path))
#English
for t in range(len(en_test)):
  
  id_=en_test['author'][t]
  lang='en'
  if en_test['pre'][t]=='1':
    type_='1'
  else:
    type_='0'
  res='<author id='+id_+'  lang='+lang+' type='+type_+' />'
  name=id_+".xml"
  with open(en_path/name, "w") as f:
    f.write(res)

#Spanish    
for t in range(len(test)):
  
  id_=test['author'][t]
  lang='es'
  if test['pre'][t]=='1':
    type_='1'
  else:
    type_='0'
  res='<author id='+id_+'  lang='+lang+' type='+type_+' />'
  name=id_+".xml"
  with open(es_path/name, "w") as f:
    f.write(res)
