from __future__ import annotations
import argparse, json
from pathlib import Path
from devspectral.config import V0
from devspectral.experiment import run_gate_suite, config_hash


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='results/v0_receipt.json')
    ap.add_argument('--check-receipt')
    args=ap.parse_args()
    if args.check_receipt:
        data=json.loads(Path(args.check_receipt).read_text())
        assert data['config_hash']==config_hash(V0), 'config hash mismatch'
        for key in ('gate0','structural_separation','probe_separation','readback','spectral_alignment','controls','lesion_regrowth','overall_status','records'):
            assert key in data, f'missing {key}'
        print('receipt OK')
        return 0
    receipt=run_gate_suite()
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(receipt.to_dict(),indent=2,sort_keys=True))
    print(json.dumps({'overall_status':receipt.overall_status,'config_hash':receipt.config_hash,'records':len(receipt.records)},indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())
