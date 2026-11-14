"""Fetch title/authors/abstract from the arXiv API by id or by title search. No API key, no personal info (generic User-Agent)."""
import sys, json, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
NS={'a':'http://www.w3.org/2005/Atom'}
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'lit-fetch/0.1'})
    return urllib.request.urlopen(req,timeout=60).read()
def parse(xml):
    out=[]
    for e in ET.fromstring(xml).findall('a:entry',NS):
        out.append(dict(id=e.find('a:id',NS).text.strip(),title=' '.join(e.find('a:title',NS).text.split()),published=e.find('a:published',NS).text[:10],
                        authors=[a.find('a:name',NS).text for a in e.findall('a:author',NS)][:6],abstract=' '.join(e.find('a:summary',NS).text.split())))
    return out
out=open(sys.argv[2],'a')
for q in open(sys.argv[1]):
    q=q.strip()
    if not q or q.startswith('#'): continue
    if q[:4].isdigit() and '.' in q[:6]: url='http://export.arxiv.org/api/query?id_list='+q
    else: url='http://export.arxiv.org/api/query?max_results=3&search_query='+urllib.parse.quote('ti:"'+q+'"')
    try: r=parse(get(url))
    except Exception as e: print('ERR',q,e); time.sleep(10); continue
    if not r: print('NONE',q)
    for w in r[:1]:
        w['query']=q; out.write(json.dumps(w)+'\n'); out.flush(); print('OK',q[:45],'->',w['title'][:70],w['published'],len(w['abstract']))
    time.sleep(4)
