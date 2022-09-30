def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value) 
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars\n
        torch.backends.cudnn.deterministic = True  #needed\n
        torch.backends.cudnn.benchmark = False
"""
def parse_args():
    parser = argparse.ArgumentParser(description = "Train Model Organ Specific for Probability Map")
    parser.add_argument('--root_dir', type = str, help = "Patch(Random Extract) Directory")
    parser.add_argument('--result_dir', type = str, help = "Save Model Parameter & Loss")
    parser.add_argument('--batch_size', type = int, default = 100)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--num_epochs', type = int, default = 100)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--level', type = int, help = "level 0 : 20X, level 1 : 5X")
    parser.add_argument('--organ', type = str, help = "'all, col', 'pan', 'pros'")
    parser.add_argument('--seed', type = int, default = 42)
    return parser.parse_args()


if __name__ == '__main__':
    global args
    args = parse_args()
    random_seed(args.seed, True)
    main()
"""