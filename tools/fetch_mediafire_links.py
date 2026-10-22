#!/usr/bin/env python3
import urllib.request
import re
import sys


def main():
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://www.mediafire.com/api/1.4/folder/get_content.php?folder_key=rw51an7tk7bh2&content_type=files&filter=in_trash=false&response_format=json"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.mediafire.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    try:
        data = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
        links = re.findall(r'href="(https://download[^"]+)"', data)
        scripts = re.findall(r'window\._static[^"]*\s*=\s*"([^"]*download[^"]*)"', data)
        direct = re.findall(r'="(https://[^"]*download[^"]*\.[^"]*)"', data)
        print("download_links:", links[:10])
        print("static_scripts:", scripts[:10])
        print("direct:", direct[:10])
        for line in data.split("\n"):
            if "download" in line.lower() and "http" in line.lower():
                print("line:", line.strip()[:300])
    except Exception as e:
        print("ERR", type(e).__name__, str(e)[:300])


if __name__ == "__main__":
    main()
