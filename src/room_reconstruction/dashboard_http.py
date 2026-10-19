"""Loopback-only HTTP adapter; streamed files, same-origin mutation endpoints."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import secrets
from urllib.parse import parse_qs, unquote, urlsplit

from .dashboard_service import BusyError


def make_server(service, *, port=8765):
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'

        def log_message(self, fmt, *args):
            # Access logs omit query strings, video filenames and tokens.
            pass

        def _headers(self, status, kind, size, extra=None):
            self.send_response(status); self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(size));self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Cache-Control','no-store');self.send_header('X-Frame-Options','SAMEORIGIN')
            self.send_header('Referrer-Policy','same-origin')
            for k,v in (extra or {}).items():self.send_header(k,v)
            self.end_headers()

        def _json(self,status,data):
            payload=json.dumps(data,ensure_ascii=False).encode()
            self._headers(status,'application/json; charset=utf-8',len(payload))
            if self.command!='HEAD':self.wfile.write(payload)

        def _guard(self,mutate=False):
            hosts={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
            host=self.headers.get('Host','')
            if host not in hosts:raise PermissionError('Host tidak diizinkan; gunakan alamat localhost ArchiCon')
            origin=self.headers.get('Origin')
            if origin and origin!='http://'+host:raise PermissionError('Origin tidak diizinkan')
            if mutate and not secrets.compare_digest(self.headers.get('X-ArchiCon-Token',''),service.token):
                raise PermissionError('Token sesi tidak valid; muat ulang dashboard')

        def _length(self,limit):
            if self.headers.get('Transfer-Encoding'):raise ValueError('Upload membutuhkan Content-Length')
            value=self.headers.get('Content-Length','')
            if not value.isdigit() or not 0<int(value)<=limit:raise ValueError('Ukuran request di luar batas')
            return int(value)

        def _body(self):
            data=json.loads(self.rfile.read(self._length(8192)))
            if not isinstance(data,dict):raise ValueError('Body JSON harus berupa object')
            return data

        def _file(self,path,*,download=False,name=None,html=False):
            path=Path(path);size=path.stat().st_size;start=0;end=size-1;status=200
            headers={'Accept-Ranges':'bytes'}
            requested=self.headers.get('Range')
            if requested:
                match=re.fullmatch(r'bytes=(\d*)-(\d*)',requested)
                if not match or not any(match.groups()):
                    self._headers(416,'text/plain',0,{'Content-Range':f'bytes */{size}'});return
                left,right=match.groups()
                if left:start=int(left);end=min(int(right),size-1) if right else size-1
                else:start=max(0,size-int(right));end=size-1
                if start>=size or start>end:
                    self._headers(416,'text/plain',0,{'Content-Range':f'bytes */{size}'});return
                status=206;headers['Content-Range']=f'bytes {start}-{end}/{size}'
            if download:
                filename=re.sub(r'[^A-Za-z0-9_.-]','_',name or path.name)
                headers['Content-Disposition']='attachment; filename="'+filename+'"'
            if html:
                # Generated room-tour HTML embeds its own scripts/data. No outbound connections.
                headers['Content-Security-Policy']="default-src 'self' data: blob:; script-src 'self' 'unsafe-inline' blob:; style-src 'self' 'unsafe-inline'; connect-src 'none'; object-src 'none'; frame-ancestors 'self'"
            kind=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
            self._headers(status,kind,max(0,end-start+1),headers)
            if self.command=='HEAD':return
            with path.open('rb') as stream:
                stream.seek(start);remaining=end-start+1
                while remaining>0:
                    block=stream.read(min(256*1024,remaining))
                    if not block:break
                    self.wfile.write(block);remaining-=len(block)

        def _dispatch(self):
            mutate=self.command=='POST';self._guard(mutate)
            parsed=urlsplit(self.path);path=unquote(parsed.path);query=parse_qs(parsed.query)
            def arg(name):
                values=query.get(name,[])
                if len(values)!=1:raise ValueError('Parameter tidak valid: '+name)
                return values[0]
            if self.command in ('GET','HEAD'):
                if path=='/api/config':
                    return self._json(200,dict(name='ArchiCon',token=service.token,max_upload_bytes=service.max_upload_bytes,control=service.control()))
                if path=='/api/inputs':return self._json(200,service.inputs())
                if path=='/api/jobs':return self._json(200,dict(jobs=service.jobs(),control=service.control()))
                if path=='/api/video':return self._file(service.input_file(arg('path')))
                match=re.fullmatch(r'/api/jobs/([A-Za-z0-9_-]+)(/artifact|/package)?',path)
                if match:
                    jid,suffix=match.groups()
                    if suffix=='/artifact':
                        file=service.artifact_file(jid,arg('path'))
                        return self._file(file,download=query.get('download')==['1'],html=file.suffix=='.html')
                    if suffix=='/package':
                        archive=service.product_zip(jid)
                        try:return self._file(archive,download=True,name=jid+'.zip')
                        finally:archive.unlink(missing_ok=True)
                    return self._json(200,service.job_view(jid))
                static={'/':'index.html','/dashboard.css':'dashboard.css','/dashboard.js':'dashboard.js'}
                if path in static:
                    file=service.root/'web/dashboard'/static[path]
                    if not file.is_file():raise FileNotFoundError('Asset dashboard belum tersedia')
                    return self._file(file)
                if re.fullmatch(r'/fonts/[A-Za-z0-9_.-]+\.(woff2|txt)',path):
                    file=service.root/'web/dashboard'/path.lstrip('/')
                    if file.is_file():return self._file(file)
                raise FileNotFoundError('Endpoint tidak ditemukan')
            if mutate:
                if path=='/api/uploads':
                    result=service.save_upload(self.rfile,self._length(service.max_upload_bytes),arg('filename'))
                    return self._json(201,result)
                if path=='/api/jobs':
                    data=self._body()
                    return self._json(201,service.start_job(data.get('input_path',''),data.get('job_id','')))
                match=re.fullmatch(r'/api/jobs/([A-Za-z0-9_-]+)/resume',path)
                if match:
                    return self._json(202,service.resume_job(match.group(1)))
            raise FileNotFoundError('Endpoint tidak ditemukan')

        def _handle(self):
            try:self._dispatch()
            except PermissionError as exc:
                self.close_connection=True;self._json(403,dict(error=str(exc)))
            except FileExistsError as exc:
                self.close_connection=True;self._json(409,dict(error=str(exc)))
            except BusyError as exc:
                self.close_connection=True;self._json(409,dict(error=str(exc)))
            except FileNotFoundError as exc:
                self.close_connection=True;self._json(404,dict(error=str(exc)))
            except (ValueError,TypeError,KeyError) as exc:
                self.close_connection=True;self._json(400,dict(error=str(exc)))
            except (BrokenPipeError,ConnectionResetError,TimeoutError):
                self.close_connection=True
            except OSError:
                self.close_connection=True;self._json(500,dict(error='Operasi file/proses gagal. Periksa ruang disk dan log lokal.'))

        do_GET=_handle
        do_HEAD=_handle
        do_POST=_handle

    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads=True
    return server
