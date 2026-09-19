# utility-scripts
Just some useful scripts.

## gitsetup.sh
Just a script to automatically create a new Github repo and push the contents of a local folder. It takes three arguments: the path to a local directory, your Github username, and the name of the new repo you want to set up.
NB: The script assumes that you have public key authentication set up for your Github account. This way you can push without authenticating.

## find-replace-perl.sh
Just a script to quickly find and replace text in files. It takes three arguments: text to find, text to replace it with, and the file name.

## scrape-md.sh
Just a script to pull the contents of a webpage and convert to Markdown with the ever-handy Pandoc utility.

## scrape-txt.sh
Just a script to pull the text from a webpage and convert it to a plain TXT file using the Lynx CLI browser.


## jev-ornith-bench/
A test harness for a specific question: does putting Jev (TypeSafe's System One model, which can only pick from typed options) in front of a local model like Ornith make it better at real sysadmin and clerical work? Three arms — the local model alone, the local model with JSON-schema-constrained decoding, and Jev — so the result can distinguish "Jev has good judgement" from "any schema helps". Runs fully offline against scripted backends with `job smoke` before you spend GPU time on it. See its own README.
