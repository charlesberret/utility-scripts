#!/usr/local/bin/bash

dir="${SCRAPE_DIR:-$HOME/notes/scraped}"   ## override with SCRAPE_DIR
date=$(date +%Y%m%d) ## date as e.g. 20160704
fulldate=$(date +'%v') ## date as e.g. 04-July-2016

## prompt for the url to scrape
echo "ENTER THE URL OF THE WEBSITE"
read url

## if the url contains a trailing slash, remove it; the naming script below
## looks for the page name after the final slash in the url
if [[ "$url" == */ ]]; then
  url="${url::-1}"
fi

## the next two lines extract the domain name, first by removing everything 
## up to and including the // in the url, e.g. it cuts "http://" et cetera
domain="${url#*//}" 
## ... and then removing everything after the next / in the remaining string
## e.g. "example.com/foo/bar/" becomes "example.com"
domain="${domain%%/*}"

## extracts the pagename, e.g. the chunk of the url after the final slash.
## nb: the function beiginning "if [[ "$url" == */ ]]" above ensures that 
## the final character is not a "/"
page=$(echo ${url##*/})

## path to directory and filename as date + domain name + page name dot markdown
filename="$dir/$date-$domain-$page.md"
echo -e "FILE WILL BE SAVED TO:\n$filename"

## pandoc reads html directly from the url
## the "+RTS -K16777216 -RTS" segment is a debug to increase buffer size
pandoc -f html -t markdown +RTS -K16777216 -RTS $url > $filename

# enter the full date and source url to the first line of whatever.md
sed -i "1s|^|Downloaded on $fulldate from $url \n\n|" $filename

## remove any line beginning with "<"
## this cleans up most HTML elements that pandoc didn't handle
sed -i -e '/^</ d' $filename

echo "FINISHED"
