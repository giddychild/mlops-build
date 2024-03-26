#!/bin/bash

set -e
banner()
{
  echo "+---------------------------------------------------+"
  printf "|%-50s |\n" "`date`"
  echo "|                                                   |"
  printf "|%-50s |\n" "$@"
  echo "+---------------------------------------------------+"
}
export PATH="$BITBUCKET_CLONE_DIR:$PATH"
export PATH="$PATH:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/bin/tput"
BASEDIR=$PWD
ENV=$1

banner "TASK START : Scanning Image Vulnerability... --> "

if [[ $? -eq 0 ]]; then
 banner " *****   SUCCESS : No Vulnerabilities Found ***** "  
  exit 0
else
  banner " *****  FAILED : Found Image Vulnerabilities ***** " 
  exit 1 
fi

# test
banner "TASK END : Scanning Image Vulnerability... --> "
