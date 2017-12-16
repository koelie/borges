"""Prepare images for training"""
import os
import argparse
import sys
import logging
import json
import uuid
import requests
import time

import numpy as np
from os.path import join, isdir, dirname, abspath

log = logging.getLogger(__name__)

def make_bom(values):
    
    with open(join(dirname(abspath(__file__)),'bom_template.json')) as f:
        template = json.load(f)
    
    types = {
        0: "flat",
        1: "smooth",
    }

    ctr1 = template["geometry"]["inputs"]["guideCurveGroup"]["curves"][0]["inputs"]["editableCurve"]["value"]["controls"]
    for i in range(7):
        ctr1.append({
            "type": types[values[i*3]],
            "x": values[i*3+1],
            "y": values[i*3+2],
        })
    
    ctr2 = template["geometry"]["inputs"]["profileCurveGroup"]["curves"][0]["inputs"]["editableCurve"]["value"]["controls"]
    for i in range(7, 23):
        ctr2.append({
            "type": types[values[i*3]],
            "x": values[i*3+1],
            "y": values[i*3+2],
        })
    return template


def get_more_data(data_path, class_name, num):
    """gets more images by randomly combining field values"""
    
    # find the class names
    classes = [fn for fn in os.listdir(data_path) if isdir(join(data_path, fn))]
    assert class_name in classes, "Given class name %s does not exist in datadir %s" % (class_name, data_path)
    
    # find the field files and join them in a big matrix
    fields_path = join(data_path, class_name, 'values')
    fields_fns = [join(fields_path, fn) for fn in os.listdir(fields_path) if fn.endswith('.fields')]
    field_list = [json.load(open(f)) for f in fields_fns]
    fields = np.array(field_list)
    nf = fields.shape[1]
    
    # generate new bom files by combining existing ones
    gen_id = "generated_{}".format(uuid.uuid4().hex[:8])
    new_boms = []
    for i in range(num):
        # combine two random objects
        a_idx, b_idx = np.random.choice(fields.shape[0], 2)
        orig_a = fields[a_idx]
        orig_b = fields[b_idx]
        values = []
        for j in range(nf):
            if j%3==0:
                # int value, pick one at random
                value = int(np.random.choice((orig_a[j], orig_b[j]), 1))
            else:
                value = float(np.mean((orig_a[j], orig_b[j])))
            values.append(value)

        if values not in field_list:
            item_id = "{}_{}".format(gen_id, uuid.uuid4().hex)
            log.info("Requesting %s from api (%d/%d)", item_id, i, num)

            val_path = join(data_path, class_name, 'values', "{}.fields".format(item_id))
            with open(val_path, 'w') as f:
                json.dump(values, f)

            bom = make_bom(values)
            bom_path = join(data_path, class_name, 'bom', "{}.bom".format(item_id))
            with open(bom_path, 'w') as f:
                json.dump(bom, f, indent=4)

            # generate a thumbnail of the model via the remote api
            png_path = join(data_path, class_name, 'png', '{}.png'.format(item_id))
            url = "https://dev.borges.xyz/api/thumbnail/?size=256"
            response = requests.post(url, data=json.dumps(bom), headers={'Content-Type': 'application/json'}, verify=False)
            assert response.status_code==201, "Bad response code from borges api: %d" % response.status_code
            with open(png_path, 'wb') as f:
                f.write(response.content)
            time.sleep(0.1)
#            cmd = 'curl -o "{}" -H "Content-Type: application/json" -X POST -d @{} https://dev.borges.xyz/api/thumbnail/?size=256 --insecure'
#            cmd = cmd.format(png_path, bom_path)
#            print(cmd)
#            os.system(cmd.format(png_path, bom_path))



if __name__ == '__main__':
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(
        description='Display ImageIterator samples')
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='enable debug logging',
    )
    parser.add_argument(
        'data_path', type=str,
        help='data directory from the assessment, with bowl and vase images',
    )
    parser.add_argument(
        'class_name', type=str, choices=('vase','bowl'),
        help='which class to get more data for'
    )
    parser.add_argument(
        'num', type=int,
        help='number of samples to get'
    )

    args = parser.parse_args()
    loglvl = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        level=loglvl,
        format='[%(asctime)s: %(levelname)s] %(message)s'
    )

    get_more_data(args.data_path, args.class_name, args.num)
