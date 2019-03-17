#!/bin/bash
folder=forVincent
mkdir $folder 
mv review.* $folder
mv user.& $folder
pushd . 
cd $folder
# Should run on all the files we just put in the folder
python ~/Downloads/convert.py
popd
