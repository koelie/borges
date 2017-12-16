"""Prepare images for training"""
import argparse
import sys
import logging
import json
import uuid
import time
import os
from os.path import join, isdir, dirname, abspath

import requests
import numpy as np

log = logging.getLogger(__name__)


def make_bom(values):
    """Make a new .bom file with given field values

    Parameters
    ----------
    values : list
        control values for the guide curve and profile curve

    Returns
    -------
    bom : dict
        bom template with input values filled in
    """
    template_fn = join(dirname(abspath(__file__)), 'bom_template.json')
    with open(template_fn) as in_file:
        template = json.load(in_file)

    types = {
        0: "flat",
        1: "smooth",
    }

    gc_curves = template["geometry"]["inputs"]["guideCurveGroup"]["curves"]
    ctr1 = gc_curves[0]["inputs"]["editableCurve"]["value"]["controls"]
    for i in range(7):
        ctr1.append({
            "type": types[values[i*3]],
            "x": values[i*3+1],
            "y": values[i*3+2],
        })

    pr_curves = template["geometry"]["inputs"]["profileCurveGroup"]["curves"]
    ctr2 = pr_curves[0]["inputs"]["editableCurve"]["value"]["controls"]
    for i in range(7, 23):
        ctr2.append({
            "type": types[values[i*3]],
            "x": values[i*3+1],
            "y": values[i*3+2],
        })
    return template


def get_more_data(data_path, class_name, num):
    """Gets more images by randomly combining field values

    Takes two existing expressions and averages them to get a new .bom file.
    Then requests a png from the borges api.

    Parameters
    ----------
    data_path : str
        path to assignment dataset
    class_name : str
        which class to get more data for
    num : int
        amount of new samples to get
    """

    # find the class names
    classes = [fn for fn in os.listdir(data_path)
               if isdir(join(data_path, fn))]
    assert class_name in classes, "Class name %s not found" % class_name

    # find the field files and join them in a big matrix
    fields_path = join(data_path, class_name, 'values')
    fields_fns = [join(fields_path, fn) for fn in os.listdir(fields_path)
                  if fn.endswith('.fields')]
    field_list = [json.load(open(fn)) for fn in fields_fns]
    fields = np.array(field_list)

    # generate new bom files by combining existing ones
    gen_id = "generated_{}".format(uuid.uuid4().hex[:8])
    for i in range(num):
        # combine two random objects
        a_idx, b_idx = np.random.choice(fields.shape[0], 2)
        orig_a = fields[a_idx]
        orig_b = fields[b_idx]
        values = []
        for j in range(fields.shape[1]):
            if j % 3 == 0:
                # int value, pick one at random
                value = int(np.random.choice((orig_a[j], orig_b[j]), 1))
            else:
                value = float(np.mean((orig_a[j], orig_b[j])))
            values.append(value)

        if values not in field_list:
            item_id = "{}_{}".format(gen_id, uuid.uuid4().hex)
            log.info("Requesting %s from api (%d/%d)", item_id, i, num)

            val_path = join(
                data_path, class_name, 'values', "{}.fields".format(item_id)
            )
            with open(val_path, 'w') as out_file:
                json.dump(values, out_file)

            bom = make_bom(values)
            bom_path = join(
                data_path, class_name, 'bom', "{}.bom".format(item_id)
            )
            with open(bom_path, 'w') as out_file:
                json.dump(bom, out_file, indent=4)

            # generate a thumbnail of the model via the remote api
            png_path = join(
                data_path, class_name, 'png', '{}.png'.format(item_id)
            )
            url = "https://dev.borges.xyz/api/thumbnail/?size=256"
            resp = requests.post(
                url, data=json.dumps(bom),
                headers={'Content-Type': 'application/json'},
                verify=False
            )
            error = "Bad api response: %d" % resp.status_code
            assert resp.status_code == 201, error
            with open(png_path, 'wb') as out_file:
                out_file.write(resp.content)
            time.sleep(0.1)


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
        'class_name', type=str, choices=('vase', 'bowl'),
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
