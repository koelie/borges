from icrawler.builtin import GoogleImageCrawler
import logging
import argparse
import sys
import glob
import os

log = logging.getLogger(__name__)

if __name__ == "__main__":
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(
        description='Download images from google images')
    parser.add_argument(
        'out_dir', type=str,
        help='directory to place images in',
    )
    parser.add_argument(
        'queries', type=str, nargs='+',
        help='query to search for',
    )
    parser.add_argument(
        '-s', '--min_size', type=int, default=128,
        help="minimum size (width or height)"
    )
    parser.add_argument(
        '-n', '--max_num', type=int, default=1000,
        help="max number of images to download"
    )

    args = parser.parse_args()
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s: %(levelname)s] %(message)s'
    )

    for query in args.queries:
        log.debug('querying %s', query)
        query_path = os.path.join(args.out_dir, query)
        google_crawler = GoogleImageCrawler(storage={'root_dir':query_path})
        google_crawler.crawl(
            keyword=query, max_num=1000,
            date_min=None, date_max=None,
            min_size=(args.min_size, args.min_size), max_size=None
        )
        for filename in os.listdir(query_path):
            orig_path = os.path.join(query_path, filename)
            out_fn = "{}/{}_{}".format(args.out_dir, query, filename.lower())
            out_fn.replace('.jpeg', '.jpg')
            if not out_fn.endswith('jpg'):
                log.warning("deleting %s", orig_path)
                os.remove(orig_path)
                continue
            log.info("Moving %s to %s", orig_path, out_fn)
            os.rename(orig_path, out_fn)
        log.info("Removing folder %s", query)
        os.rmdir(query_path)
