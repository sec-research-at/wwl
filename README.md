# wwl
Faster "CeWL" replacement by Sec-Research 
Author: Schrader 


wwl (world wide list) is a wordlist generator that scrapes an URI to create a wordlist which then can be used for attacks on credentials.


## Installation

```
git clone https://github.com/sec-research-at/wwl.git
pip install -r requirements.txt
python3 wwl.py -h
```

## Usage
```
usage: wwl.py [-h] [--threads THREADS] [--output OUTPUT] [--insecure] [--add-lowercase] [--alphabet ALPHABET]
              [--user-agent USER_AGENT] [--min-word-length MIN_WORD_LENGTH] [--header HEADER] [--max-depth MAX_DEPTH]
              url

positional arguments:
  url                   Url to crawl and extract keywords from

options:
  -h, --help            show this help message and exit
  --threads THREADS     Number of threads used for processing
  --output OUTPUT, -o OUTPUT
                        File to write keywords to (default stdout)
  --insecure, -k        Allow insecure SSL connections and transfers.
  --add-lowercase, -al  Add lowercase variants of all keywords
  --alphabet ALPHABET, -a ALPHABET
                        Alphabet that is used to clean the edges of the keywords
  --user-agent USER_AGENT, -ua USER_AGENT
                        User agent to use for requests
  --min-word-length MIN_WORD_LENGTH, -mwl MIN_WORD_LENGTH
                        Minimum word length
  --header HEADER, -H HEADER
                        Header in the format of name:value
  --max-depth MAX_DEPTH, -md MAX_DEPTH
                        Maximum depth to crawl (-1 = no limit)
```
