# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 23:51:49 2020

@author: user
"""

import argparse
import os  
import codecs
import sys

def read_text(path):
    with codecs.open(path , 'r' , encoding="utf-8") as f:
        return f.read()

def prepare_dataset(input_dir, truth_path):
    truth_file = read_text(truth_path)
    train_fake, train_genuine, fakeid, genid = [], [], [], [] 
    
    for xml_file in truth_file.split('\n'):
        if len(xml_file) <= 1:
            continue
        xml_filepath = os.path.join(input_dir , xml_file.split(":::")[0] + '.xml')
        content = open(xml_filepath, encoding = "utf-8").read()
        
        tweets = []
        i = 0
        while True:
            i += 1
            print(" i :   ", i)
            start_documents = content.find('<document>')
            end_documents = content.find('</document>')
            tweets.append(' '.join(content[start_documents + 19 : end_documents-3].split()))
            content = content[end_documents+10:]
            if i == 100:
                break
        if xml_file.split(":::")[1] == '1':
            train_fake.append(' '.join(tweets))
            fakeid.append(xml_file.split(":::")[0])
        else:
            train_genuine.append(' '.join(tweets))
            genid.append(xml_file.split(":::")[0])
            
    return train_fake, train_genuine, fakeid, genid

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="path to input dataset")
    parser.add_argument('-o', '--output', help="path to output directory", default='prepared_dataset')
    args = parser.parse_args()
    if args.input is None:
        parser.print_usage()
        sys.exit()
    return args

def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print('directory created')

def main():
    args = get_args()
    input_dir = os.path.normpath(args.input)
    out = os.path.normpath(args.output)
    mkdir(out)
    
    for dir in os.listdir(input_dir):
        print("Working on Language:", dir )
        out_dir = os.path.join(out, dir)
        mkdir(out_dir)
        truthfile_path = os.path.join(input_dir, dir, 'truth.txt') 
        
        fake, genuine, fakeid, genid = prepare_dataset(os.path.join(input_dir, dir), truthfile_path)
        with codecs.open(os.path.join(out_dir,'train_fake.txt'),'w' , encoding='utf-8' ) as f:
                f.write('\n'.join(fake))
        with codecs.open(os.path.join(out_dir,'train_genuine.txt'),'w' , encoding='utf-8' ) as f:
                f.write('\n'.join(genuine))
        with codecs.open(os.path.join(out_dir,'fakeid.txt'),'w' , encoding='utf-8' ) as f:
                f.write('\n'.join(fakeid))
        with codecs.open(os.path.join(out_dir,'genid.txt'),'w' , encoding='utf-8' ) as f:
                f.write('\n'.join(genid))
        
        print("Fake train-set size: ", len(fake), "Fakeid size: ", len(fakeid))
        print("Genuine train-set size: ", len(genuine), "genid size: ", len(genid))
        print("Dataset saved to ", str(out_dir))
        print("--------------------------------------------------")

main()
