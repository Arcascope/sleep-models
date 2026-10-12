"""Durable local completion/failure events; a separate Codex task relays them to the parent."""
import argparse,hashlib,json,os,time
from datetime import datetime,timezone
from pathlib import Path

EXCLUDE={'checks','nights','data','bundles','bundle','models','diagnostics'}


def read(path,default=None):
    try:
        if path.stat().st_size>262144:return default
        return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError,OSError):return default


def save(path,data):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(data,indent=2));os.replace(tmp,path)


def ident(kind,path,stamp):
    return hashlib.sha256(f'{kind}:{path}:{stamp}'.encode()).hexdigest()[:24]


def candidates(root):
    for run in root.glob('boat-*'):
        if not run.is_dir() or run.is_symlink():continue
        for filename in ('config.json','complete.json','collection-complete.json'):
            for file in run.rglob(filename):
                relative=file.relative_to(run)
                if set(relative.parts)&EXCLUDE:continue
                yield file


def serve(a):
    a.state.mkdir(parents=True,exist_ok=True)
    lock=a.state/'daemon.pid'
    if lock.exists():
        try:os.kill(int(lock.read_text()),0)
        except (ProcessLookupError,ValueError):pass
        else:raise RuntimeError('A monitor daemon is already running')
    lock.write_text(str(os.getpid()))
    state=read(a.state/'seen.json',{})
    # A fresh installation acknowledges historical completions/configs instead of spamming them.
    if not state:
        for f in candidates(a.root):state[ident(f.name,str(f),f.stat().st_mtime_ns)]='historical'
        save(a.state/'seen.json',state)
    deadline=time.time()+a.hours*3600
    events=a.state/'events.jsonl'
    def emit(kind,path,stamp,**extra):
        key=ident(kind,str(path),stamp)
        if key in state:return
        event=dict(id=key,kind=kind,path=str(path.relative_to(a.root)),utc=datetime.now(timezone.utc).isoformat(),**extra)
        with events.open('a') as out:out.write(json.dumps(event)+'\n')
        state[key]=kind;save(a.state/'seen.json',state)
    save(a.state/'status.json',dict(pid=os.getpid(),state='armed',deadline_utc=datetime.fromtimestamp(deadline,timezone.utc).isoformat(),interval_seconds=a.interval))
    print('monitor armed',flush=True)
    while time.time()<deadline:
        for f in candidates(a.root):
            value=read(f)
            if not isinstance(value,dict):continue
            stamp=f.stat().st_mtime_ns
            if f.name!='config.json':
                key=ident(f.name,str(f),stamp)
                if key in state:continue
                # Give downstream scorer time to flush files; notification says marker, not proven gain.
                if time.time()-f.stat().st_mtime<20:continue
                emit(f.name,f,stamp);continue
            if 'pid' not in value or 'aws' in f.parts:continue
            key=ident('config.json',str(f),stamp)
            if state.get(key)=='historical':continue
            done=f.parent/'complete.json'
            if done.exists() and done.stat().st_mtime>=f.stat().st_mtime:continue
            if time.time()-f.stat().st_mtime<90:continue
            try:os.kill(int(value['pid']),0)
            except ProcessLookupError:emit('failed',f,stamp,reason='Local training process exited without a fresh completion marker')
            except (ValueError,PermissionError):continue
        request=read(a.state/'self-test-request.json')
        if request:
            emit('self_test',a.state/'self-test-request.json',request['id'])
        time.sleep(a.interval)
    emit('monitor_expired',a.state/'status.json',str(deadline))
    save(a.state/'status.json',dict(pid=os.getpid(),state='expired',utc=datetime.now(timezone.utc).isoformat()))
    print('monitor expired',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('state',type=Path)
    p.add_argument('--hours',type=float,default=8);p.add_argument('--interval',type=float,default=15)
    serve(p.parse_args())
