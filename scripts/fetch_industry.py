"""Download the raw industry-classification data used by scripts/build_relations.py.

  csrc     CSRC industry, point-in-time: one baostock snapshot per month end (record effective from `updateDate`)
           -> <out>/csrc_pit_raw_<start>.csv
  cninfo   industry-classification change history of every stock (all standards, incl. Shenwan old / 2021) from the
           cninfo p_stock2110 API -> <out>/cninfo_raw/<instrument>.json   (resumable: existing files are skipped)
  snapshot current CSRC industry of every stock (baostock latest) -> data/stock_industry.csv ('snap' control graph)

The stock list is the union of instruments in the MASTER opensource datasets.
Usage:
  python scripts/fetch_industry.py csrc --start 2007-12-01 --end 2022-12-31
  python scripts/fetch_industry.py cninfo
  python scripts/fetch_industry.py snapshot --out_csv data/stock_industry.csv
Requires: baostock, cryptography (network access to baostock.com / webapi.cninfo.com.cn).
"""
import argparse
import base64
import json
import os
import pickle
import random
import time
import urllib.parse
import urllib.request

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('what', choices=['csrc', 'cninfo', 'snapshot'])
p.add_argument('--data_dir', default='/data/dataset/MASTER/opensource')
p.add_argument('--out', default='/data/dataset/MASTER_relations')
p.add_argument('--start', default='2007-12-01')
p.add_argument('--end', default='2022-12-31')
p.add_argument('--out_csv', default='data/stock_industry.csv')
args = p.parse_args()


def instruments():
    insts = set()
    for split in ['train', 'valid', 'test']:
        for u in ['csi300', 'csi800']:
            ds = pickle.load(open(f'{args.data_dir}/{u}_dl_{split}.pkl', 'rb'))
            insts |= set(ds.get_index().get_level_values('instrument').unique())
    return sorted(insts)


def baostock_query(bs, **kw):
    """query_stock_industry with re-login on session expiry (error 10001001), which otherwise silently returns
    empty months during long downloads."""
    for attempt in range(5):
        rs = bs.query_stock_industry(**kw)
        rows = []
        while rs.error_code == '0' and rs.next():
            rows.append(rs.get_row_data())
        if rs.error_code == '0':
            return rows, rs.fields
        print('baostock', kw, 'error', rs.error_code, rs.error_msg, '-> re-login', flush=True)
        bs.logout()
        time.sleep(2 + 3 * attempt)
        bs.login()
    raise RuntimeError(f'baostock query failed: {kw}')


def fetch_csrc():
    import baostock as bs
    bs.login()
    rows, fields = [], None
    for d in pd.date_range(args.start, args.end, freq='ME'):
        r, fields = baostock_query(bs, date=d.strftime('%Y-%m-%d'))
        rows += [x + [d.strftime('%Y-%m-%d')] for x in r]
        print(d.date(), len(r), flush=True)
    bs.logout()
    df = pd.DataFrame(rows, columns=fields + ['query_date'])
    df['instrument'] = df['code'].str.replace('.', '', regex=False).str.upper()
    os.makedirs(args.out, exist_ok=True)
    out = f'{args.out}/csrc_pit_raw_{args.start}.csv'
    df.to_csv(out, index=False)
    print('written', out, len(df), 'records,', df['updateDate'].nunique(), 'effective dates')


def cninfo_enckey():
    """Accept-Enckey header of webapi.cninfo.com.cn: AES-128-CBC (key = iv = '1234567887654321', PKCS7) of the
    current unix time, base64 (same as the site's JS / akshare)."""
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    k = b'1234567887654321'
    pad = padding.PKCS7(128).padder()
    data = pad.update(str(int(time.time())).encode()) + pad.finalize()
    enc = Cipher(algorithms.AES(k), modes.CBC(k)).encryptor()
    return base64.b64encode(enc.update(data) + enc.finalize()).decode()


def cninfo_query(code, sdate='2005-01-01', edate='2022-12-31'):
    url = 'https://webapi.cninfo.com.cn/api/stock/p_stock2110?' + urllib.parse.urlencode({'scode': code, 'sdate': sdate, 'edate': edate})
    req = urllib.request.Request(url, data=b'', method='POST', headers={
        'Accept-Enckey': cninfo_enckey(), 'Origin': 'https://webapi.cninfo.com.cn', 'Referer': 'https://webapi.cninfo.com.cn/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest'})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def fetch_cninfo():
    out = f'{args.out}/cninfo_raw'
    os.makedirs(out, exist_ok=True)
    insts = instruments()
    print('instruments', len(insts), flush=True)
    for i, inst in enumerate(insts):
        fn = f'{out}/{inst}.json'
        if os.path.exists(fn):
            continue
        for attempt in range(5):
            try:
                r = cninfo_query(inst[2:])
                assert r.get('resultcode') == 200, r.get('resultmsg')
                json.dump(r.get('records', []), open(fn, 'w'), ensure_ascii=False)
                break
            except Exception as e:  # network hiccups / rate limiting
                print(inst, 'retry', attempt, e, flush=True)
                time.sleep(5 + 5 * attempt)
        if i % 100 == 0:
            print(time.strftime('%H:%M:%S'), i, inst, flush=True)
        time.sleep(0.3 + 0.2 * random.random())


def fetch_snapshot():
    import baostock as bs
    bs.login()
    rows, fields = baostock_query(bs)
    bs.logout()
    df = pd.DataFrame(rows, columns=fields)
    df['instrument'] = df['code'].str.replace('.', '', regex=False).str.upper()
    df = df[df.instrument.isin(set(instruments()))][['instrument', 'code_name', 'industry', 'industryClassification', 'updateDate']]
    df.to_csv(args.out_csv, index=False)
    print('written', args.out_csv, len(df))


{'csrc': fetch_csrc, 'cninfo': fetch_cninfo, 'snapshot': fetch_snapshot}[args.what]()
