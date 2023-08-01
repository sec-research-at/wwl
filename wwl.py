#!/usr/bin/env python3

from argparse import ArgumentParser
from urllib.parse import urlparse, urljoin
from threading import Thread, Lock
from queue import Queue, Empty
from requests import Session, packages, Response
from bs4 import BeautifulSoup
from time import sleep, time
import re
from urllib3.exceptions import InsecureRequestWarning
from rich.console import Console
import sys
import os.path
from string import ascii_letters, digits, punctuation
import multiprocessing

DEFAULT_ALPHABET = ascii_letters + digits + "äöüÄÖÜß-"
DEFAULT_SPLIT_CHARS = punctuation.replace("-", "") + " \t"


parser = ArgumentParser()
parser.add_argument(
    "input", type=str, help="Url or filename (containing list of urls) to crawl and extract keywords from", nargs="+")
parser.add_argument("--threads", type=int, default=multiprocessing.cpu_count(),
                    help="Number of threads used for processing")
parser.add_argument("--output", "-o", type=str,
                    help="File to write keywords to (default stdout)")
parser.add_argument("--insecure", "-k", action="store_true",
                    help="Allow insecure SSL connections and transfers.")
parser.add_argument("--lowercase", "-l", action="store_true",
                    help="Add lowercase variants of all keywords", dest="add_lowercase")
parser.add_argument("--alphabet", "-a", default=DEFAULT_ALPHABET,
                    help="Alphabet that is used to clean the edges of the keywords")
parser.add_argument("--user-agent", "-ua",
                    help="User agent to use for requests")
parser.add_argument("--min-word-length", "-miwl",
                    help="Minimum word length", default=3, type=int)
parser.add_argument("--max-word-length", "-mawl",
                    help="Maximum word length", default=30, type=int)
parser.add_argument(
    "--header", "-H", help="Header in the format of name:value", action="append")
parser.add_argument("--max-depth", "-md", default=-1,
                    help="Maximum depth to crawl (-1 = no limit)", type=int)
parser.add_argument("--split-chars", "-sc", default=DEFAULT_SPLIT_CHARS)
parser.add_argument("--force-base-url", "-fbp", action="store_true",
                    help="Only crawl urls that start with the supplied input url(s)")

console = Console(file=sys.stderr)


args = parser.parse_args()
base_urls = []

for input in args.input:
    if os.path.exists(input):
        with open(input, "r") as f:
            lines = f.read().splitlines()
            for url in [urlparse(line) for line in lines]:
                base_urls.append(url)
    else:
        url = urlparse(input)
        base_urls.append(url)

for base_url in base_urls:
    if not base_url.scheme or not base_url.netloc:
        console.print(f"[red][-] Invalid url: {base_url.geturl()}[/red]")
        exit(-1)

base_url_strs = [url.geturl() for url in base_urls]

if args.insecure:
    packages.urllib3.disable_warnings(category=InsecureRequestWarning)

split_regex = "("+"|".join(["\\" + x for x in list(args.split_chars)])+")"

visited = set()  # urls visited by crawler
visited_lock = Lock()
queue = Queue()  # url queue
# urls not to be added to queue again - need separate set as queue is not hashed
queue_locked_urls = set()
queue_lock = Lock()
busy = Queue()  # when a thread processes a new url it adds to the queue, when it's done it removes - used to keep track of busy threads

keywords = set()  # unique extracted keywords
kw_queues = [Queue() for _ in range(args.threads)]

cancel_signal = False

for url in base_url_strs:
    queue.put((0, url))


def new_session():
    """creates new requests session using cli args"""
    session = Session()

    if args.user_agent:
        session.headers["User-Agent"] = args.user_agent

    if args.header:
        for header in args.header:
            name, value = header.split(":")
            session.headers[name] = value

    if args.insecure:
        # allow insecure ssl
        session.verify = False

    return session


def queue_if_relevant(base, url, depth):
    """queue an url if it is relevant"""

    if args.max_depth > -1 and depth + 1 > args.max_depth:
        # too deep
        return

    url = urlparse(urljoin(base, url, allow_fragments=False))
    url_str = url.geturl()

    found = False
    for base_url in base_urls:
        if base_url.scheme == url.scheme and base_url.netloc == url.netloc:
            found = True
            break

    if not found:
        return

    if url.fragment is not None and len(url.fragment) > 0:
        # remove fragments from urls
        url_str = url_str[:-len(url.fragment)]

    if args.force_base_url:
        found = False
        for base_url in base_url_strs:
            if url_str.startswith(base_url):
                found = True
                break
        if not found:
            # url is not in our base
            return

    if url_str in visited:
        # already visited
        return

    try:
        queue_lock.acquire()
        if url_str in queue_locked_urls:
            return
        queue_locked_urls.add(url_str)
        queue.put((depth+1, url_str))
    finally:
        queue_lock.release()


def clean_keyword(kw: str):
    """clean a keyword based on our policy"""
    if not kw:
        return None

    if not any(filter(lambda x: x in args.alphabet, kw)):
        # no character in the keyword is valid
        return None

    # get index of first valid char in keyword
    start = 0
    for start in range(0, len(kw)):
        if kw[start] in args.alphabet:
            break

    # get index of last valid char in keyword
    end = len(kw)-1
    for end in range(len(kw)-1, 0, -1):
        if kw[end] in args.alphabet:
            break

    # trim keyword based on indexes
    kw = kw[start:end+1]

    if len(kw) < args.min_word_length or len(kw) > args.max_word_length:
        return None

    return kw

def parse_html(res: Response, depth: int, analyse):
    soup = BeautifulSoup(res.content, features="html.parser")

    for a in soup.find_all("a", href=True):
        # queue all <a hrefs>
        href = a["href"]
        queue_if_relevant(url, href, depth)

    for tag in soup.find_all():
        if tag.name in ["script", "style"]:
            continue
        if tag.string is None:
            continue
        val = tag.string
        if "style" in tag and val == tag["style"]\
                or "class" in tag and val == tag["class"]:
            continue
        val = val.replace("\n", "")
        if not val:
            continue

        analyse(val)


parsers = {
    r"text\/html.+": parse_html
}

def add_worker(id):
    """add a new worker node"""
    session = new_session()

    def analyse(text):
        """extract keywords from text node"""
        for word in re.split(split_regex, text):
            word = clean_keyword(word)
            if not word:
                continue

            # add keyword to queue
            kw_queues[id].put(word)
            if args.add_lowercase:
                kw_queues[id].put(word.lower())

    def process(data):
        """process a queued url"""

        depth, url = data

        try:
            visited_lock.acquire()
            if url in visited:
                # in case of race condition with visited set, can just be ignored
                return
            visited.add(url)
        finally:
            visited_lock.release()

        try:
            queue_lock.acquire()
            if url in queue_locked_urls:
                queue_locked_urls.remove(url)
        finally:
            queue_lock.release()

        response = session.get(url)
        content_type = response.headers["content-type"].lower().strip()

        # find a parser and run it
        for (regex, parser) in parsers:
            if re.match(regex, content_type):
                parser(response, depth, analyse)
                return


    def work():
        """Process queue items until empty"""
        # busy queue is used to keep the program alive until all nodes are done and there is nothing more queued
        is_busy = False
        while not cancel_signal:
            try:
                data = queue.get(timeout=1)
                busy.put(id)
                is_busy = True
                process(data)
                queue.task_done()
            except BaseException as e:
                if not isinstance(e, Empty):
                    console.print(
                        f"[red][e{id}] {type(e).__name__} occured: {str(e)}[/red]")

            if is_busy:
                busy.get(block=False)
                busy.task_done()

            is_busy = False

    return work


workers = [Thread(target=add_worker(i), daemon=True)
           for i in range(args.threads)]

start = time()

for worker in workers:
    worker.start()

file = open(args.output, "w") if args.output else sys.stdout


def get_status_str():
    tt = round(time() - start, 2)
    return f"{tt}s  {len(keywords)} keywords, {queue.qsize()} urls in queue, {busy.qsize()} threads busy"


def process_kw_queue(max=None):
    """processes thread-safe keyword queues from nodes and adds them to a set"""
    i = 0
    for qi in range(len(kw_queues)):
        curr_queue = kw_queues[qi]
        while (not max or i < max) and not curr_queue.empty():
            keyword = curr_queue.get()
            if keyword not in keywords:
                keywords.add(keyword)
                file.write(f"{keyword}\n")
            i += 1
    file.flush()


with console.status(get_status_str()) as status:
    try:
        while not queue.empty() or not busy.empty():
            process_kw_queue(max=1000)
            status.update(get_status_str())
            sleep(0.1)
    except KeyboardInterrupt:
        pass

    cancel_signal = True

    status.update("Waiting for workers to complete...")

    for worker in workers:
        worker.join()

process_kw_queue()

tt = round(time() - start, 2)

console.print(
    f"[green][+] Found {len(keywords)} keywords in [bold]{tt}s[/bold] on {len(visited)} sites.[/green]")

file.close()
