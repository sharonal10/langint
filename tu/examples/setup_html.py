# This is an example script for setting up HTML visualizations.
import numpy as np
from tu.loggers.utils import setup_vi
from PIL import Image
from tu.common import ROOT
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


def main():
    # The following line sets up a HTML webpage.
    # Double-click the URL printed in the console to open in the browser.
    vi, vi_helper = setup_vi(path=Path(ROOT) / 'logs' / 'setup_html', title='My HTML')

    im_arr = np.ones((128, 128, 3), dtype=np.uint8)

    vi_helper.dump_table(
        vi,
        layout=[
            [
                dict(
                    image=Image.fromarray(im_arr * 0),
                    info='The first row is a black image.',
                )
            ],
            [
                dict(
                    image=Image.fromarray(im_arr * 128),
                    info='The second row is a grey image.',
                )
            ]
        ],
        table_name='My images with texts',
        col_type='auto',
        col_names=['my-column'],
    )

    vi_helper.dump_table(
        vi,
        layout=[
            [Image.fromarray(im_arr * 0) for _ in range(2)],
            [Image.fromarray(im_arr * 0) for _ in range(2)],
        ],
        table_name='My table of 2 x 2 images',
        col_type='auto',
        col_names=None,
    )


if __name__ == "__main__":
    main()
