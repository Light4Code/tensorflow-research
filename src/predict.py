import argparse


def main():
  parser = argparse.ArgumentParser(
        description='Used to predict TensorFlow model checkpoint')
  parser.add_argument(
        'config',
        metavar="config",
        help='Path to the configuration file containing all parameters for model training'
    )    

if __name__ == '__main__':
    main()
