#! /usr/bin/env perl
use warnings;
use strict;

use Getopt::Long;
use Pod::Usage;

#############
# Variables #
#############
my $dir = ".";
my $dryrun = 0;
my $help = 0;
my $substitutor = "./substitutor.sh";
my $verbose = 0;

my $template_file = undef;
my $output_prefix = undef;

##########
# Getopt #
##########
if (!GetOptions("dir|d=s" => \$dir,
		"dry-run|n" => \$dryrun,
		"help|h" => \$help,
		"substitutor|s=s" => \$substitutor,
		"verbose|v" => \$verbose)) {
  die("Error in command-line options");
}

###############
# Error Check #
###############
if ($help) {
  pod2usage(-verbose => 2, -exitval => 0);
  exit 0;
}

if (scalar(@ARGV) < 2) {
  print STDERR "Input template and output basename not specified!\n";
  pod2usage(-verbose => 0, -exitval => 1, -output => \*STDERR);
  exit 1;
} else {
  $template_file = $ARGV[0];
  $output_prefix = $ARGV[1];
}

while (my $line = <STDIN>) {
  chomp $line;

  if ($line =~ /^([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)$/) {
    my $terminals = $1;
    my $time = $2;
    my $rate = $3;
    my $weights = $4;

    my $output_suffix =
      $terminals . "_" . $time. "_" . $rate. "_" . $weights . ".xml";

    my $commandline =
      "$substitutor " .
      "--terminals=$terminals " .
      "--time=$time " .
      "--rate=$rate " .
      "--weights=$weights " .
      "$template_file " .
      "> $dir/$output_prefix" . "_" . "$output_suffix";

    if ($verbose || $dryrun) {
      print "$commandline\n";
    }

    if (!$dryrun) {
      if (0 != system("$commandline")) {
	print STDERR "\"$commandline\" failed. Skipping\n";
      }
    }
  } else {
    print STDERR "\"$line\" is not a valid parameter set! Skipping\n";
  }
}

=head1 NAME

generator.pl - Generate oltpbench configuration files

=head1 SYNOPSIS

generator.pl [-d|--dir=DIRECTORY] [-n|--dry-run] [-h|--help]
[-s|--substitutor=SUBSTITUTOR] [-v|--verbose] TEMPLATE_FILE
OUTPUT_PREFIX

=head1 OPTIONS

-d, --dir=DIRECTORY Prepend DIRECTORY to output file names

-n, --dry-run Output commands to be run, but do not run
 them. Overrides --verbose.

-h, --help Print this help text and exit

-s, --substitutor=SUBSTITUTOR Use SUBSTITUTOR instead of
 C<./substitutor.sh>

-v, --verbose Output commands being run.

=head1 ARGUMENTS

TEMPLATE_FILE: The template file to use to create the configuration
files

OUTPUT_PREFIX: The prefix to the output file

=head1 DESCRIPTION

After taking in the options, generator.pl reads STDIN to get a list of
parameter sets. Each parameter set should be separated by a newline. A
parameter set has the format:

 <TERMINALS> <TIME> <RATE> <WEIGHTS>

These parameters are in the format as for substitutor.sh and
white-space separated.

generator.pl generates a configuration file from the specified
TEMPLATE_FILE for each parameter set and outputs the file of the
format

  <OUTPUT_PREFIX>_<TERMINALS>_<TIME>_<RATE>_<WEIGHTS>.xml

in the current working directory or in DIRECTORY if specified.

=cut
