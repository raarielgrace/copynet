import argparse
import torch

def main(model_path):
    model = torch.load('./' + model_path + '.pt') # Reload model just in case
    torch.save(model.state_dict(), model_path[model_path.rfind('/') + 1:] + '_cpu.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse training parameters')
    parser.add_argument('model_path', type=str,
                        help='the name of the model, without the extension')

    args = parser.parse_args()
    main(args.model_path)
