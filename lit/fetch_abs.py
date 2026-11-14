"""Fetch title/venue/abstract from OpenAlex for a list of title queries (one per line). No API key, no personal info."""
import sys, json, urllib.request, urllib.parse, time
def inv(a):
    if not a: return ''
    pos=[]
    for w,ps in a.items():
        for p in ps: pos.append((p,w))
    return ' '.join(w for _,w in sorted(pos))
out=open(sys.argv[2],'a')
for q in open(sys.argv[1]):
    q=q.strip()
    if not q or q.startswith('#'): continue
    url='https://api.openalex.org/works?per-page=3&select=id,title,publication_year,primary_location,authorships,abstract_inverted_index,cited_by_count,doi&search='+urllib.parse.quote(q)
    try:
        r=json.load(urllib.request.urlopen(url,timeout=40))['results']
    except Exception as e:
        print('ERR',q,e); time.sleep(20); continue
    # prefer result with abstract
    r=sorted(r,key=lambda x: x.get('abstract_inverted_index') is None)
    if not r: print('NONE',q); continue
    w=r[0]
    rec=dict(query=q,title=w['title'],year=w['publication_year'],venue=((w.get('primary_location') or {}).get('source') or {}).get('display_name'),
             authors=[a['author']['display_name'] for a in w['authorships'][:4]],cites=w['cited_by_count'],doi=w.get('doi'),abstract=inv(w.get('abstract_inverted_index')))
    out.write(json.dumps(rec)+'\n'); out.flush()
    print('OK',q[:50],'->',(w['title'] or '')[:70],w['publication_year'],len(rec['abstract']))
    time.sleep(6)
