#! /bin/bash

cd /home/ec2-user/SageMaker/Coswara-Data
if [ -d ".git" ]; then
  rm -fr .git
fi
for file in *; do
  if [ -d "$file" ]; then
    echo "Processing folder: $file"
    cd $file
    cat *.tar.gz.* > $file.tar.gz
    tar -xzf $file.tar.gz
    rm *.tar.gz*
    cd $file
    echo "number of annotated samples is `ls -hl | wc -l`"
    for subfile in *; do
        if [ -d "$subfile" ]; then
            cd $subfile
            status=`grep covid_status metadata.json | awk '{i=index($0,"covid_status");split(substr($0,i),a,"\"");print a[3]}'`
            for wavfile in *; do
                filesize=$(stat -c%s "$wavfile")
                case "$wavfile" in
                    "breathing-deep.wav") echo "$file/$file/$subfile/$wavfile,$filesize,$status" >> ../../../breathing-deep-metadata.csv 
                    ;;
                    "breathing-shallow.wav") echo "$file/$file/$subfile/$wavfile,$filesize,$status" >> ../../../breathing-shallow-metadata.csv 
                    ;;
                    "cough-heavy.wav") echo "$file/$file/$subfile/$wavfile,$filesize,$status" >> ../../../cough-heavy-metadata.csv 
                    ;;
                    "cough-shallow.wav") echo "$file/$file/$subfile/$wavfile,$filesize,$status" >> ../../../cough-shallow-metadata.csv 
                    ;;
                    "counting-fast.wav") echo "$file/$file/$subfile/$wavfile,$filesize,$status" >> ../../../counting-fast-metadata.csv 
                    ;;
                    "counting-normal.wav") echo "$file/$file/$subfile/$wavfile,$filesize,$status" >> ../../../counting-normal-metadata.csv 
                    ;;
                    "vowel-a.wav") echo "$file/$file/$subfile/$wavfile,$filesize,$status" >> ../../../vowel-a-metadata.csv 
                    ;;
                    "vowel-e.wav") echo "$file/$file/$subfile/$wavfile,$filesize,$status" >> ../../../vowel-e-metadata.csv 
                    ;;
                    "vowel-o.wav") echo "$file/$file/$subfile/$wavfile,$filesize,$status" >> ../../../vowel-o-metadata.csv 
                esac
            done
            cd ..
        fi
    done
    cd ../..
  fi
done
