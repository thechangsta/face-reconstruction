import gdown
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download file from Google Drive')
    parser.add_argument('--output', type=str, help='Output file name', default='../dataset/VGGface2_HQ.zip')
    args = parser.parse_args()

    url = "https://drive.google.com/uc?id=1cPn7GRUX3150VOTkfuL4kVJIXnUdQgn6"
    gdown.download(url, args.output, quiet=False)
