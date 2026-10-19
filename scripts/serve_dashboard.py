#!/usr/bin/env python3
"""Start ArchiCon's local input/progress/output dashboard."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    from room_reconstruction.dashboard_service import DashboardService
    from room_reconstruction.dashboard_http import make_server
    service=DashboardService(ROOT);server=make_server(service,port=args.port)
    service.start_monitor()
    print(f'ArchiCon → http://127.0.0.1:{server.server_port}',flush=True)
    print('Review manual · output end_product/<job_id>/vNNN · Ctrl+C menutup web, bukan worker.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:service.stop_monitor();server.server_close()


if __name__=='__main__':main()
