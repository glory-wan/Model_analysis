
from utils.computing_function import computing_info

from model.conv import *
from model.block import *

if __name__ == '__main__':
    model = C2fCIB(c1=3, c2=64)

    computing_info(
        images_dir="./images",
        output_dir='./vis_result',
        model_name='C2fCIB',
        device='cpu',
        model=model,
    )
