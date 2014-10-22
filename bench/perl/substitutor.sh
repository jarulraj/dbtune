#! /usr/bin/env bash

#############
# Variables #
#############
TERMINALS_PARAMETER='%TERMINALS%'
TIME_PARAMETER='%TIME%'
RATE_PARAMETER='%RATE%'
WEIGHTS_PARAMETER='%WEIGHTS%'

TERMINALS="$TERMINALS_PARAMETER"
TIME="$TIME_PARAMETER"
RATE="$RATE_PARAMETER"
WEIGHTS="$WEIGHTS_PARAMETER"

##############
# getoptions #
##############
GETOPT=$(getopt --options="he:t:r:w:" --longoptions="help,terminals:,time:,rate:,weights:" -n "generator.sh" -- "$@")
if [[ $? != 0 ]]; then
    exit 1;
fi

eval set -- "$GETOPT"

while true; do
    case "$1" in
	-h|--help)
	    cat <<EOF
Usage: substitutor.sh [OPTION]... TEMPLATE_FILE
Write TEMPLATE_FILE with parameters replaced by parameter values to standard output.

Mandatory arguments to long options are mandatory for short options too.
Parameter options:
-e, --terminals=TERMINALS replace 'terminals' parameter with TERMINALS
-t, --time=TIME           replace 'time' parameter with TIME
-r, --rate=RATE           replace 'rate' parameter with RATE
-w, --weights=WEIGHTS     replate 'weights' parameter with WEIGHTS

Other options:
-h, --help show this help text
EOF
	    exit 0
	    ;;

	-e|--terminals)
	    TERMINALS="$2"
	    shift
	    shift
	    ;;

	-t|--time)
	    TIME="$2"
	    shift
	    shift
	    ;;

	-r|--rate)
	    RATE="$2"
	    shift
	    shift
	    ;;

	-w|--weights)
	    WEIGHTS="$2"
	    shift
	    shift
	    ;;

	--)
	    shift
	    break
	    ;;

	*)
	    echo "$1" 1>&2
	    shift
	    echo "Internal Error!" 1>&2
	    exit 1
	    ;;
    esac
done

########################
# Perform Substitution #
########################

if [[ $# == 0 ]] ; then
    echo "No template file specified!" 1>&2
    exit 1
fi

TEMPLATE_FILE="$1"

sed -e "s/$TERMINALS_PARAMETER/$TERMINALS/g" \
    -e "s/$TIME_PARAMETER/$TIME/g" \
    -e "s/$RATE_PARAMETER/$RATE/g" \
    -e "s/$WEIGHTS_PARAMETER/$WEIGHTS/g" \
    "$TEMPLATE_FILE"
