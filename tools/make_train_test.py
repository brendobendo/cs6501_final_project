#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 23:11:07 2017

@author: babraham
"""
import os
from random import random

##############Data Directory (Change this to your directory)#################
data_dir = "/home/ningjy/jianyusu/Image_recognition/cs6501_final_project/"
#############################################################################

def main():
    data_folder = data_dir + "all_data"
    os.chdir(data_folder)
    os.system("mkdir train")
    os.system("mkdir val")
    tr_prop = .75
    classes = os.listdir("images")
    train_list, val_list = [], []
    if ".DS_Store" in classes: classes.remove(".DS_Store")
    for c in classes:
        os.system("mkdir train/" + c)
        os.system("mkdir val/" + c)
        files = os.listdir("images/" + c)
        if ".DS_Store" in files: files.remove(".DS_Store")
        for f in files:
            old = "images/" + c + "/" + f
            r = random()
            if r < tr_prop:
                new = "train/" + c + "/" + f
                train_list.append(new)
            else:
                new = "val/" + c + "/" + f
                val_list.append(new)
            os.system("mv " + old + " " + new)
    datalists = [("train_list", train_list), ("val_list",val_list)]
    for i in range(2):
        fname, data = datalists[i]
        with open(fname + ".txt", 'w') as out:
            for d in data: out.write(d + "\n")
    os.system('rm -R images')
            
if __name__ == '__main__':
    main()
