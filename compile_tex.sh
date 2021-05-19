#!/usr/bin/bash
# for each lecture .tex file, compiles the .tex file with pdflatex and writes
# the output to the same directory the source .tex file is located in.

# pdflatex compile command. for some reason, -output-directory is ignored.
PDF_TEX="pdflatex -interaction=nonstopmode -halt-on-error -shell-escape"
# directory this file is located in, the repo root (relative path)
REPO_ROOT=$(dirname $0)
# base lesson directory
LESSON_ROOT=$REPO_ROOT/lessons
# error message when there are too many arguments
TOO_MANY_ARGS="$0: too many arguments. try $0 --help for usage"
# script usage
USAGE="
usage: $0 [TEXFILE] [-v]

For each .tex file in the top-level package directory and for each
lecture_[0-9]+ directory located in the top-level package directory, compiles
the .tex file to PDF with pdflatex + bibtex and writes the output to the same
directory the respective .tex file is located in. The pdflatex command used is

$PDF_TEX

The typical pdflatex -> bibtex -> pdflatex -> pdflatex chain of commands is run
to produce the PDF, which will be in the same directory as the .tex file.

If an argument is passed to this script, assumed to be a .tex file, then only
that specified file will be compiled into a PDF.

optional arguments:
 -h, --help     show this usage
 TEXFILE        specified .tex file to compile, output written to the same
                directory as TEXFILE. if omitted, then all .tex files in the
                top-level package directory and each lecture_[0-9]+ directory
                within in the top-level package directory are compiled.
 -v, --verbose  pass to show the full output of the pdflatex/bibtex commands.
                by default, the output of pdflatex/bibtex is suppressed."

# single-file compilation function. arguments:
# $1: path to .tex file
# $2: 0 for silence (direct output to /dev/null), 1 for no output suppression
compile_tex() {
    # save current directory so we can cd back
    BASE_DIR=$(pwd)
    # save file name, without extension
    LONG_PROJ_NAME=$1
    LONG_PROJ_NAME=${LONG_PROJ_NAME%%.tex}
    # save pathless file name, without extension (delete from back of string)
    PROJ_NAME=$(basename $1)
    PROJ_NAME=${PROJ_NAME%%.tex}
    # so that pdflatex doesn't screw up looking for images, cd to dir of arg.
    # note sure why pdflatex doesn't respect the \graphicspath command.
    cd $(dirname $1)
    ## build project (pdflatex -> bibtex -> pdflatex -> pdflatex) ##
    # if verbose, allow pdflatex and bibtex output to be printed to screen
    if [[ $2 == 1 ]]
    then
        echo "building $LONG_PROJ_NAME.pdf..."
        $PDF_TEX $PROJ_NAME
        bibtex $PROJ_NAME
        $PDF_TEX $PROJ_NAME
        $PDF_TEX $PROJ_NAME
        echo "done"
    # else suppress all stdout from pdflatex and bibtex
    else
        echo -n "building $LONG_PROJ_NAME.pdf..."
        $PDF_TEX $PROJ_NAME > /dev/null
        bibtex $PROJ_NAME > /dev/null
        $PDF_TEX $PROJ_NAME > /dev/null
        $PDF_TEX $PROJ_NAME > /dev/null
        echo "done"
    fi
    # return to base directory
    cd $BASE_DIR
}

# main compilation loop. arguments:
# $1: 0 for silence (direct output to /dev/null), 1 for no output suppression
compile_loop() {
    # compile all .tex files in the top-level lessons directory
    for INFILE in $LESSON_ROOT/*.tex
    do
        compile_tex $INFILE $1
    done
    # for each lecture directory in the lessons directory
    for LEC_DIR in $LESSON_ROOT/lecture_{00..15}
    do
        # compile each .tex file in the directory
        for INFILE in $LEC_DIR/*.tex
        do
            # don't feed nonexistent names to compilation function
            if [ -e $INFILE ]
            then
                compile_tex $INFILE $1
            fi
        done
    done
}

# if no arguments, then compile all lectures non-verbosely
if [[ $# == 0 ]]
then
    # compile all .tex files in the repo, suppressing pdflatex/bibtex output
    compile_loop 0
# else if 1 argument
elif [[ $# == 1 ]]
then
    # if help argument, then print usage
    if [ $1 = "-h" ] || [ $1 = "--help" ]
    then
        # need double quote to preserve the spacing
        echo "$USAGE"
    # else if -v or --verbose, don't suppress pdflatex/bibtex output when
    # compiling all the .tex files to pdf
    elif [ $1 = "-v" ] || [ $1 = "--verbose" ]
    then
        compile_loop 1
    # else treat as .tex file and send to pdflatex. output written in same dir.
    else
        compile_tex $1 0
    fi
# else if two 2 arguments
elif [[ $# == 2 ]]
then
    # if the second argument is -v or --verbose, compile file verbosely
    if [ $2 = "-v" ] || [ $2 = "--verbose" ]
    then
        compile_tex $1 1
    # else too many arguments
    else
        echo $TOO_MANY_ARGS
    fi
# else too many arguments
else
    echo $TOO_MANY_ARGS
fi