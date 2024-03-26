#!/bin/bash

COMMIT_ID=$1

# Getting list of files changed and storing it in a file
git show --pretty="format:" --name-only $COMMIT_ID > git_output.txt

# Using sed to parse the file and fetch only the app folder names
cat git_output.txt | sed -e 's|^apps\(/[^/]*/\).*$|\1|;tx;d;:x' | sed "s#^/\(.*\)#\1#" | sed "s#\(.*\)/\$#\1#" > git_output.txt

if [[ -n "$PIPELINE_APP_NAME" ]]
then
   echo "$PIPELINE_APP_NAME" > git_output.txt
fi

#Sorting and removing the duplicates to avoid multiple triggers of pipeline
sort -u git_output.txt > files_changed.txt

rm git_output.txt

# Checking if an app is changed
if [ $(wc -l < "files_changed.txt") -eq 0 ]; then
    echo "No apps changed"
    exit 0
fi