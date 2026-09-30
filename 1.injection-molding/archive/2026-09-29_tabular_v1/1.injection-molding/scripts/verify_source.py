"""Verify local CSV bytes against the official contest ZIP (no extraction)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

PROJECT=Path(__file__).resolve().parents[1]
NOTICE='https://www.kamp-ai.kr/noticeDetail?NOTICE_SEQ=86&page=1&GROUP_SEL=&SEARCH_SEL=&SEARCH_TXT='


def digest(data): return hashlib.sha256(data).hexdigest()


def main(download=False, output=None):
    archive=PROJECT/'dataset/source_archive.zip'
    doc=PROJECT/'dataset/source_task.hwpx'
    downloaded=None
    if download:
        import requests
        from bs4 import BeautifulSoup
        session=requests.Session()
        session.headers['User-Agent']='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/140.0.0.0 Safari/537.36'
        response=session.get(NOTICE,timeout=30);response.raise_for_status()
        soup=BeautifulSoup(response.text,'html.parser')
        found={}
        for a in soup.find_all('a',onclick=True):
            label=a.get_text(' ',strip=True)
            match=re.search(r"doDown\('([0-9]+)'\)",a['onclick'])
            if not match: continue
            if '1. 사출성형기 AI 데이터셋.zip' in label:found['archive']=match[1]
            if '제6회 K-인공지능 제조데이터 분석 경진대회 과제공개' in label and '.hwpx' in label:found['document']=match[1]
        if set(found)!={'archive','document'}:raise ValueError('Official attachment links changed')
        for kind,path in [('archive',archive),('document',doc)]:
            response=session.post('https://www.kamp-ai.kr/fileDownload',data={'fileSeq':found[kind]},headers={'Referer':NOTICE},timeout=60)
            response.raise_for_status()
            import io
            if not zipfile.is_zipfile(io.BytesIO(response.content)):raise ValueError('Attachment is not ZIP/HWPX')
            path.write_bytes(response.content)
        downloaded=datetime.now(timezone.utc).isoformat()
    if not archive.exists():raise FileNotFoundError('Run verify_source.py --download first')
    report={'notice_url':NOTICE,'checked_at_utc':datetime.now(timezone.utc).isoformat(),
            'downloaded_at_utc':downloaded,'archive_sha256':digest(archive.read_bytes()),'files':[],
            'label_semantics':'unconfirmed'}
    with zipfile.ZipFile(archive) as z:
        report['archive_members']=z.namelist()
        for kind in ('labeled','unlabeled'):
            for group in ('cn7','rg3'):
                name=f'moldset_{kind}_{group}.csv'
                matches=[n for n in z.namelist() if Path(n).name==name]
                if len(matches)!=1:raise ValueError(f'Ambiguous/missing official CSV: {name}')
                official=z.read(matches[0]);local=(PROJECT/'dataset'/name).read_bytes()
                report['files'].append({'file':name,'official_sha256':digest(official),'local_sha256':digest(local),'bytes_match':official==local,'bytes':len(local)})
    report['all_csv_bytes_match']=all(x['bytes_match'] for x in report['files'])
    if doc.exists():
        with zipfile.ZipFile(doc) as z:
            text=' '.join(' '.join(ET.fromstring(z.read(n)).itertext()) for n in z.namelist() if n.startswith('Contents/section') and n.endswith('.xml'))
        report['task_document_sha256']=digest(doc.read_bytes())
        report['task_document_term_present']={t:t in text for t in ['PassOrFail','CN7','RG3','라벨','정규화','표준화']}
        report['document_check_limit']='XML text keyword search only; absence of keywords is not proof about all possible metadata sources.'
    output=output or PROJECT/'artifacts/eda_v1/source_check.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['all_csv_bytes_match']:raise ValueError('Official/local data mismatch')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args();main(a.download,a.output)
