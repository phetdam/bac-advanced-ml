#!/usr/bin/bash
# for each lecture .tex file, compiles the .tex file with pdflatex and writes
# the output to the same directory the source .tex file is located in.

# pdflatex compile command. for some reason, -output-directory is ignored.
# therefore, i used -jobname to modify the output directory, which works for me.
PDF_TEX="pdflatex -interaction=nonstopmode -halt-on-error -shell-escape"
# script usage
USAGE="usage: $0 [-h] [TEXFILE]

For each .tex file in the top-level package directory and for each
lecture_[0-9]+ directory located in the top-level package directory, compiles
the .tex file with pdflatex and writes the output to the same directory the
respective .tex file is located in. The pdflatex command used is

$PDF_TEX

If an argument is passed to this script, assumed to be a .tex file, then only
that file will be compiled with output written to the same directory that the
provided .tex source file resides in.

optional arguments:
 -h, --help  show this usage
 TEXFILE     specified .tex file to compile, writing output to the same
             directory the source file is located in. if omitted, then all .tex
             files in the top-level package directory and each lecture_[0-9]+
             directory within in the top-level package directory are compiled."
# directory this file is located in, the repo root (relative path)
REPO_ROOT=$(dirname $0)

# if no arguments, then compile all lectures
if [[ $# == 0 ]]
then
    # for all .tex files in the top-level package directory
    for INFILE in $REPO_ROOT/bac_advanced_ml/*.tex
    do
        # note redirect to /dev/null doesn't include 2>&1 so output from
        # stderr will still be shown. echo + sed used to replace .tex with
        # empty string for the job name.
        $PDF_TEX -jobname="$(echo $INFILE | sed s/.tex//g)" $INFILE > /dev/null
    done
    # for each lecture directory in the package directory
    for LEC_DIR in $REPO_ROOT/bac_advanced_ml/lecture*
    do
        # for each .tex file in the directory
        for INFILE in $LEC_DIR/*.tex
        do
            # use same compilation command as above
            $PDF_TEX -jobname="$(echo $INFILE | sed s/.tex//g)" $INFILE \
                > /dev/null
        done
    done
# else if 1 argument
elif [[ $# == 1 ]]
then
    # if help argument, then print usage
    if [ $1 = "-h" ] || [ $1 = "--help" ]
    then
        # need double quote to preserve the spacing
        echo "$USAGE"
    # else treat as .tex file and send to pdflatex. output written in same dir.
    else
        $PDF_TEX -jobname="$(echo $1 | sed s/.tex//g)" $1 > /dev/null
    fi
# else too many arguments
else
    echo "$0: too many arguments. try $0 --help for usage"
fi