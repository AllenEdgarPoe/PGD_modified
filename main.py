import argparse
import json
import os
from pgd_train import *
import pathlib

def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial Attack Implementation by JKshark")
    parser.add_argument("--mode", default="pgd_train", type=str)
    args = parser.parse_args()
    return args

def main(args):
    configf_path = "./configs/pgd_training.json"
    with open(configf_path, "r") as f:
        configs = json.load(f)
    argparse_dict = vars(args)
    argparse_dict.update(configs)

    args.epsilon = float(args.epsilon)/255
    args.alpha = float(args.alpha)/255

    args.save_path = os.path.join(args.save_path, args.mode, args.model, args.dataset, args.train_attacker+"_"+args.test_attacker)
    pathlib.Path(args.save_path).mkdir(parents=True, exist_ok=True)

    if args.mode == "pgd_train":
        m = PGD_train(args)
        m.train()
        #Evaluation
        #nat_acc = m.eval_nat()
        #adv_acc = m.eval_adv()
        #m._log('Natural Accuracy: {:.3f}'.format(nat_acc))
        #m._log('Adv Accuracy: {:.3f}'.format(adv_acc))




if __name__ == '__main__':
    args = parse_args()
    main(args)