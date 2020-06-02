#!/bin/sh
#PBS -q N
#PBS -N 01
#PBS -e /srv01/xenon/gerak/logs/01.err
#PBS -o /srv01/xenon/gerak/logs/01.out
#PBS -l nodes=1:ppn=1,mem=1500mb
#PBS -l cput=6:00:00
#PBS -m n

echo Start job at `date` at `hostname`
source ~/zshrc.sh

cd ~/try/
echo Switched to $PWD
echo Running python...
python3 try.py a
echo Done.
echo End job at `date` at `hostname`
