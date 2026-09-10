"""Inventory installed distribution license metadata; unknown is not permission."""
import argparse
import importlib.metadata as metadata
from pathlib import Path
from volleymole.common import digest, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    records = {}
    for distribution in metadata.distributions():
        data = distribution.metadata
        files = []
        for relative in distribution.files or []:
            name = str(relative)
            if '.dist-info/' not in name or not any(k in name.upper() for k in ('LICENSE','COPYING','NOTICE')):
                continue
            path = Path(distribution.locate_file(relative))
            if path.is_file():
                files.append({'distribution_path':name,'sha256':digest(path)})
        license_text = data.get('License-Expression') or data.get('License') or None
        item = {'name':data['Name'],'version':distribution.version,
                        'license_metadata':license_text,
                        'classifiers':[c for c in data.get_all('Classifier',[]) if c.startswith('License ::')],
                        'notice_files':files,'unknown':not bool(license_text or files)}
        key = (data['Name'].lower().replace('_','-'),distribution.version)
        # Editable installs expose both source egg-info and installed dist-info.
        if key not in records or len(files)>len(records[key]['notice_files']):
            records[key] = item
    save_json(args.output,{'note':'Installed metadata inventory, not a legal clearance or a weight-license grant.',
        'distributions':sorted(records.values(),key=lambda r:r['name'].lower())})
    print(f'Inventoried {len(records)} installed distributions; output: {args.output}')


if __name__=='__main__':
    main()
