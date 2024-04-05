import argparse

def get_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Define hyper-parameters for the experiment.')

    # Add arguments
    parser.add_argument('--version', type=str, default='exp1', help='Version number of this experiment')
    parser.add_argument('--train_path', type=str, default='data/train_modified2.csv', help='Path to the train image list csv')
    parser.add_argument('--vali_path', type=str, default='data/vali_modified2.csv', help='Path to the validation image list csv')
    parser.add_argument('--test_path', type=str, default='data/vali_modified2.csv', help='Path to the test image list csv')
    parser.add_argument('--fc_path', type=str, default='data/downloaded_test_fc.csv', help='Path to save the feature layer values of the test data')
    parser.add_argument('--test_ckpt_path', type=str, default='cache/logs_v3_9/min_model.ckpt-27280', help='Checkpoint to load when testing')
    parser.add_argument('--ckpt_path', type=str, default='logs_v3_10/model.ckpt-59999', help='Checkpoint to load when continue training')
    parser.add_argument('--weight_decay', type=float, default=0.00025, help='Scale for l2 regularization')
    parser.add_argument('--fc_weight_decay', type=float, default=0.00025, help='Scale for fully connected layer\'s l2 regularization')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--continue_train_ckpt', action='store_true', help='Whether to continue training from a checkpoint')
    parser.add_argument('--num_residual_blocks', type=int, default=2, help='Number of residual blocks in ResNet')
    parser.add_argument('--is_localization', action='store_true', help='Add localization task or not')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')

    # Parse the arguments
    args = parser.parse_args()
    return args